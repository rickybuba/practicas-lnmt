from conllu_token import Token
from algorithm import ArcEager, Sample, Transition
from state import State
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, Input

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64, 
                 epochs: int = 1, batch_size: int = 64):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        # Vocabularies with special tokens
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.ROOT = "ROOT"
        
        self.word2id = {self.PAD: 0, self.UNK: 1, self.ROOT: 2}
        self.tag2id = {self.PAD: 0, self.UNK: 1}
        self.deprel2id = {self.PAD: 0} # 0 reserved for None/PAD
        self.id2deprel = {0: self.PAD}

        # Action mapping
        self.actions = [ArcEager.SHIFT, ArcEager.REDUCE, ArcEager.LA, ArcEager.RA]
        self.action2id = {act: i for i, act in enumerate(self.actions)}
        self.id2action = {i: act for i, act in enumerate(self.actions)}

        self.model = None
        self.arc_eager = ArcEager()

    def _extract_features(self, samples, fit_vocab=False):
        """Helper to extract features and vectorize samples."""
        X_word, X_tag, Y_act, Y_dep = [], [], [], []

        for sample in samples:
            # Extract raw string features [word_s1, word_s0, word_b0, word_b1, upos_s1...]
            # Using 2 stack + 2 buffer features as per guide examples
            feats = sample.state_to_feats(nbuffer_feats=2, nstack_feats=2)
            n_feats = len(feats)
            half = n_feats // 2
            words = feats[:half]
            tags = feats[half:]

            # Fit vocabularies if training
            if fit_vocab:
                for w in words:
                    if w not in self.word2id: self.word2id[w] = len(self.word2id)
                for t in tags:
                    if t not in self.tag2id: self.tag2id[t] = len(self.tag2id)
                
                dep = sample.transition.dependency
                if dep and dep not in self.deprel2id:
                    idx = len(self.deprel2id)
                    self.deprel2id[dep] = idx
                    self.id2deprel[idx] = dep

            # Vectorize Inputs
            w_ids = [self.word2id.get(w, self.word2id[self.UNK]) for w in words]
            t_ids = [self.tag2id.get(t, self.tag2id[self.UNK]) for t in tags]
            
            X_word.append(w_ids)
            X_tag.append(t_ids)

            # Vectorize Outputs
            Y_act.append(self.action2id[sample.transition.action])
            dep_label = sample.transition.dependency
            Y_dep.append(self.deprel2id.get(dep_label, 0)) # 0 if None

        return np.array(X_word), np.array(X_tag), np.array(Y_act), np.array(Y_dep)
    
    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.
        """
        # 1. Prepare Data
        print("Vectorizing training data...")
        XW_train, XT_train, YA_train, YD_train = self._extract_features(training_samples, fit_vocab=True)
        print("Vectorizing dev data...")
        XW_dev, XT_dev, YA_dev, YD_dev = self._extract_features(dev_samples, fit_vocab=False)

        print(f"Vocab sizes - Words: {len(self.word2id)}, Tags: {len(self.tag2id)}, Deprels: {len(self.deprel2id)}")

        # 2. Define Model (Functional API)
        # Inputs: 4 words features, 4 tag features
        in_words = Input(shape=(4,), name='input_words')
        in_tags = Input(shape=(4,), name='input_tags')

        # Embeddings
        emb_words = layers.Embedding(len(self.word2id), self.word_emb_dim)(in_words)
        emb_tags = layers.Embedding(len(self.tag2id), 50)(in_tags) # Tag dim fixed to 50 usually enough

        # Flatten & Concat
        flat_words = layers.Flatten()(emb_words)
        flat_tags = layers.Flatten()(emb_tags)
        merged = layers.Concatenate()([flat_words, flat_tags])

        # Hidden Layer
        hidden = layers.Dense(self.hidden_dim, activation='relu')(merged)

        # Outputs (Multi-task)
        out_act = layers.Dense(4, activation='softmax', name='act_out')(hidden) # 4 transitions
        out_dep = layers.Dense(len(self.deprel2id), activation='softmax', name='dep_out')(hidden)

        self.model = models.Model(inputs=[in_words, in_tags], outputs=[out_act, out_dep])
        
        # 3. Compile & Fit
        # CORRECCIÓN AQUÍ: Definimos métricas específicas para cada salida usando un diccionario
        self.model.compile(optimizer='adam',
                           loss={'act_out': 'sparse_categorical_crossentropy', 
                                 'dep_out': 'sparse_categorical_crossentropy'},
                           metrics={'act_out': 'accuracy', 
                                    'dep_out': 'accuracy'})
        
        self.model.fit(
            {'input_words': XW_train, 'input_tags': XT_train},
            {'act_out': YA_train, 'dep_out': YD_train},
            validation_data=(
                {'input_words': XW_dev, 'input_tags': XT_dev},
                {'act_out': YA_dev, 'dep_out': YD_dev}
            ),
            epochs=self.epochs,
            batch_size=self.batch_size
        )

    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """
        XW, XT, YA, YD = self._extract_features(samples, fit_vocab=False)
        
        results = self.model.evaluate(
            {'input_words': XW, 'input_tags': XT},
            {'act_out': YA, 'dep_out': YD},
            verbose=1
        )
        
        uas = results[3] # act_out_accuracy
        las = results[4] # dep_out_accuracy

        print(f"\nEvaluation Results:")
        print(f"Action Accuracy (UAS-ish): {uas*100:.2f}%")
        print(f"Deprel Accuracy (LAS-ish): {las*100:.2f}%")
        
        return uas, las
    
    def run(self, sents: list['Token']):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                 as a list of Token objects.
        """

        # Main Steps for Processing Sentences:
        # 1. Initialize: Create the initial state for each sentence.
        # We also keep track of original sentence index to update tokens later if needed,
        # but here we modify state.A which uses IDs, so we can map back easily.
        states = [self.arc_eager.create_initial_state(s) for s in sents]
        
        # Keep list of active states (not yet final)
        active_indices = list(range(len(states)))

        # 8. Iterative Process
        while active_indices:
            
            # Prepare batch for current active states
            current_states = [states[i] for i in active_indices]
            XW_batch, XT_batch = [], []

            # 2. Feature Representation
            for state in current_states:
                # Create a dummy sample just to reuse state_to_feats logic
                dummy_sample = Sample(state, Transition(ArcEager.SHIFT))
                feats = dummy_sample.state_to_feats(nbuffer_feats=2, nstack_feats=2)
                
                n = len(feats) // 2
                w_ids = [self.word2id.get(w, self.word2id[self.UNK]) for w in feats[:n]]
                t_ids = [self.tag2id.get(t, self.tag2id[self.UNK]) for t in feats[n:]]
                XW_batch.append(w_ids)
                XT_batch.append(t_ids)

            # 3. Model Prediction
            preds = self.model.predict(
                {'input_words': np.array(XW_batch), 'input_tags': np.array(XT_batch)}, 
                verbose=0
            )
            act_probs = preds[0] # [batch, 4]
            dep_probs = preds[1] # [batch, n_deps]

            # Lists to decide which states remain active
            next_active_indices = []

            for i, state_idx in enumerate(active_indices):
                state = states[state_idx]
                
                # 4. Transition Sorting
                # Get action indices sorted by probability descending
                sorted_act_ids = np.argsort(act_probs[i])[::-1]
                
                # Get best dependency label
                best_dep_id = np.argmax(dep_probs[i])
                best_dep_label = self.id2deprel.get(best_dep_id, "_")

                # 5. Validation Check
                transition_applied = False
                for act_id in sorted_act_ids:
                    action_str = self.id2action[act_id]
                    
                    is_valid = False
                    if action_str == ArcEager.LA: is_valid = self.arc_eager.LA_is_valid(state)
                    elif action_str == ArcEager.RA: is_valid = self.arc_eager.RA_is_valid(state)
                    elif action_str == ArcEager.REDUCE: is_valid = self.arc_eager.REDUCE_is_valid(state)
                    elif action_str == ArcEager.SHIFT: is_valid = len(state.B) > 0 # SHIFT valid if buffer not empty

                    if is_valid:
                        # 6. State Update
                        # Dep label is only needed for LA/RA
                        final_dep = best_dep_label if action_str in [ArcEager.LA, ArcEager.RA] else None
                        trans = Transition(action_str, final_dep)
                        self.arc_eager.apply_transition(state, trans)
                        transition_applied = True
                        break
                
                # Force finish if stuck (failsafe)
                if not transition_applied:
                    continue 

                # 7. Final State Check
                if not self.arc_eager.final_state(state):
                    next_active_indices.append(state_idx)

            active_indices = next_active_indices

        # Post-process: Update the original Token objects with the predicted arcs
        for i, sent in enumerate(sents):
            state = states[i]
            # Reset
            for token in sent:
                if token.id != 0:
                    token.head = "_"
                    token.dep = "_"
            
            # Apply arcs: (head_id, dep_label, dependent_id)
            for (h, d_lbl, d_id) in state.A:
                # Access token by index (assuming tokens are ordered 0..N)
                # Token with id=d_id is at index d_id
                if d_id < len(sent):
                    sent[d_id].head = h
                    sent[d_id].dep = d_lbl


if __name__ == "__main__":
    
    model = ParserMLP()