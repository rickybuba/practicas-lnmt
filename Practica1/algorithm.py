from state import State
from conllu_token import Token


class Transition(object):
    """
    Class to represent a parsing transition in a dependency parser.
    
    Attributes:
    - action (str): The action to take, represented as an string constant. Actions include SHIFT, REDUCE, LEFT-ARC, or RIGHT-ARC.
    - dependency (str): The type of dependency relationship (only for LEFT-ARC and RIGHT-ARC, otherwise it'll be None), corresponding to the deprel column
    """

    def __init__(self, action: int, dependency: str = None):
        self._action = action
        self._dependency = dependency

    @property
    def action(self):
        """Return the action attribute."""
        return self._action

    @property
    def dependency(self):
        """Return the dependency attribute."""
        return self._dependency

    def __str__(self):
        return f"{self._action}-{self._dependency}" if self._dependency else str(self._action)


class Sample(object):
    """
    Represents a training sample for a transition-based dependency parser. 

    This class encapsulates a parser state and the corresponding transition action 
    to be taken in that state. It is used for training models that predict parser actions 
    based on the current state of the parsing process.
    """

    def __init__(self, state: State, transition: Transition):
        self._state = state
        self._transition = transition

    @property
    def state(self):
        return self._state

    @property
    def transition(self):
        return self._transition
    

    def state_to_feats(self, nbuffer_feats: int = 2, nstack_feats: int = 2):
            """
            Extracts features from a given parsing state for use in a transition-based dependency parser.
            [cite_start][cite: 18-22] - Feature extraction logic based on stack and buffer.
            """
            # List to store word forms and UPOS tags
            words_stack = []
            words_buffer = []
            upos_stack = []
            upos_buffer = []

            # 1. Extract features from Stack (Order: deeper -> top)
            # We want the top 'nstack_feats' elements.
            stack_len = len(self.state.S)
            
            # Iteramos desde nstack_feats hacia abajo (ej: 2, 1) para obtener el orden [S-2, S-1]
            for i in range(nstack_feats, 0, -1):
                idx = stack_len - i
                if idx >= 0:
                    token = self.state.S[idx]
                    words_stack.append(token.form)
                    upos_stack.append(token.upos)
                else:
                    words_stack.append("<PAD>")
                    upos_stack.append("<PAD>")

            # 2. Extract features from Buffer (Order: front -> back)
            # We want the first 'nbuffer_feats' elements.
            buffer_len = len(self.state.B)
            
            for i in range(nbuffer_feats):
                if i < buffer_len:
                    token = self.state.B[i]
                    words_buffer.append(token.form)
                    upos_buffer.append(token.upos)
                else:
                    words_buffer.append("<PAD>")
                    upos_buffer.append("<PAD>")

            # Concatenate lists in the specified order: 
            # [Word_stack, Word_buffer, UPOS_stack, UPOS_buffer]
            return words_stack + words_buffer + upos_stack + upos_buffer
    

    def __str__(self):
        return f"Sample - State:\n\n{self._state}\nSample - Transition: {self._transition}"


class ArcEager():

    """
    Implements the arc-eager transition-based parsing algorithm for dependency parsing.

    This class includes methods for creating initial parsing states, applying transitions to 
    these states, and determining the correct sequence of transitions for a given sentence.

    Methods:
        create_initial_state(sent: list[Token]): Creates the initial state for a given sentence.
        final_state(state: State): Checks if the current parsing state is a valid final configuration.
        LA_is_valid(state: State): Determines if a LEFT-ARC transition is valid for the current state.
        LA_is_correct(state: State): Determines if a LEFT-ARC transition is correct for the current state.
        RA_is_correct(state: State): Determines if a RIGHT-ARC transition is correct for the current state.
        RA_is_valid(state: State): Checks if a RIGHT-ARC transition is valid for the current state.
        REDUCE_is_correct(state: State): Determines if a REDUCE transition is correct for the current state.
        REDUCE_is_valid(state: State): Determines if a REDUCE transition is valid for the current state.
        oracle(sent: list[Token]): Computes the gold transitions for a given sentence.
        apply_transition(state: State, transition: Transition): Applies a given transition to the current state.
        gold_arcs(sent: list[Token]): Extracts gold-standard dependency arcs from a sentence.
    """

    LA = "LEFT-ARC"
    RA = "RIGHT-ARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def __init__(self):
        # Helper storage for oracle correctness checks
        self.gold_set = None

    def create_initial_state(self, sent: list['Token']) -> State:
        """
        Creates the initial state for the arc-eager parsing algorithm given a sentence.

        This function initializes the parsing state, which is essential for beginning the parsing process. 
        The initial state consists of a stack (initially containing only the root token), a buffer 
        (containing all tokens of the sentence except the root), and an empty set of arcs.

        Parameters:
            sent (list[Token]): A list of 'Token' instances representing the sentence to be parsed. 
                                The first token in the list should typically be a 'ROOT' token.

        Returns:
            State: The initial parsing state, comprising a stack with the root token, a buffer with 
                the remaining tokens, and an empty set of arcs.
        """
        return State([sent[0]], sent[1:], set([]))
    
    def final_state(self, state: State) -> bool:
        """
        Checks if the curent parsing state is a valid final configuration, i.e., the buffer is empty

            Parameters:
                state (State): The parsing configuration to be checked

            Returns: A boolean that indicates if state is final or not
        """
        return len(state.B) == 0

    # --- Helper to check if a node has a head in the current arcs ---
    def _has_head(self, node_id: int, arcs: set) -> bool:
        for (h, dep, d) in arcs:
            if d == node_id:
                return True
        return False
    
    # --- Helper to get the dependency label from gold arcs ---
    def _get_gold_dep(self, head_id: int, child_id: int) -> str:
        if self.gold_set is None: return None
        for (h, dep, d) in self.gold_set:
            if h == head_id and d == child_id:
                return dep
        return None

    def LA_is_valid(self, state: State) -> bool:
        """
       
        LEFT-ARC Conditions:
        1. Stack is not empty (implied by getting S[-1], but checked for safety).
        2. Buffer is not empty (implied by context).
        3. Top of stack (i) is not ROOT (0).
        4. Top of stack (i) does not have a head yet.
        """
        if not state.S or not state.B:
            return False
        
        top_stack = state.S[-1]
        
        # Condition: i != 0 (ROOT)
        if top_stack.id == 0:
            return False
        
        # Condition: i does not have a head
        if self._has_head(top_stack.id, state.A):
            return False
            
        return True

    def LA_is_correct(self, state: State) -> bool:
        """
        [cite: 259] - Correct if there is a link from Buffer[0] (j) to Stack[-1] (i) in Gold.
        """
        if self.gold_set is None: 
            raise RuntimeError("Oracle not initialized with gold arcs.")
        
        i = state.S[-1].id
        j = state.B[0].id
        
        # Check if gold arc (j -> i) exists
        for (h, dep, d) in self.gold_set:
            if h == j and d == i:
                return True
        return False
    
    def RA_is_correct(self, state: State) -> bool:
        """
        [cite: 263] - Correct if there is a link from Stack[-1] (i) to Buffer[0] (j) in Gold.
        """
        if self.gold_set is None: 
            raise RuntimeError("Oracle not initialized with gold arcs.")
        
        i = state.S[-1].id
        j = state.B[0].id
        
        # Check if gold arc (i -> j) exists
        for (h, dep, d) in self.gold_set:
            if h == i and d == j:
                return True
        return False

    def RA_is_valid(self, state: State) -> bool:
        """
       
        RIGHT-ARC Conditions:
        1. Stack is not empty.
        2. Buffer is not empty.
        3. Top of Buffer (j) does not have a head yet.
        """
        if not state.S or not state.B:
            return False
        
        top_buffer = state.B[0]
        
        # Condition: j does not have a head
        if self._has_head(top_buffer.id, state.A):
            return False
            
        return True

    def REDUCE_is_correct(self, state: State) -> bool:
        """
        [cite: 238, 257] + Standard Arc-Eager Oracle Logic.
        REDUCE is correct if:
        1. It is valid (Stack[-1] has a head).
        2. Stack[-1] does NOT have any children in the Buffer (or anywhere else not yet attached).
           Since we parse left-to-right, we strictly check if Stack[-1] has any children in the Buffer.
        """
        if not self.REDUCE_is_valid(state):
            return False
            
        i = state.S[-1].id
        
        # Check if 'i' is the head of any node currently in the Buffer (or Stack, though unlikely in this algo)
        # If 'i' still has children waiting in the buffer, we cannot REDUCE yet (we need to wait to RA them).
        for token in state.B:
            # Check if exists gold arc (i -> token)
            for (h, dep, d) in self.gold_set:
                if h == i and d == token.id:
                    return False # Found a child in buffer, cannot reduce
        
        return True

    def REDUCE_is_valid(self, state: State) -> bool:
        """
       
        REDUCE Conditions:
        1. Stack is not empty.
        2. Top of Stack (i) has a head.
        """
        if not state.S:
            return False
        
        top_stack = state.S[-1]
        
        # Condition: i has a head
        return self._has_head(top_stack.id, state.A)

    def oracle(self, sent: list['Token']) -> list['Sample']:
        """
        Computes the gold transitions to take at each parsing step.
        """
        self.gold_set = self.gold_arcs(sent)

        state = self.create_initial_state(sent) 
        samples = [] 

        # Applies the transition system until a final configuration state is reached
        while not self.final_state(state):
            
            state_copy = State(list(state.S), list(state.B), set(state.A))

            if self.LA_is_valid(state) and self.LA_is_correct(state):
                label = state.S[-1].dep
                transition = Transition(self.LA, label)
                samples.append(Sample(state_copy, transition))
                self.apply_transition(state, transition)

            elif self.RA_is_valid(state) and self.RA_is_correct(state):
                label = state.B[0].dep
                transition = Transition(self.RA, label)
                samples.append(Sample(state_copy, transition))
                self.apply_transition(state, transition)

            elif self.REDUCE_is_valid(state) and self.REDUCE_is_correct(state):
                transition = Transition(self.REDUCE)
                samples.append(Sample(state_copy, transition))
                self.apply_transition(state, transition)
                
            else:
                transition = Transition(self.SHIFT)
                samples.append(Sample(state_copy, transition))
                self.apply_transition(state, transition)

        # Verificación final
        assert self.gold_arcs(sent) == state.A, f"Gold arcs and generated arcs do not match"
        
        # Limpieza (opcional, pero buena práctica)
        self.gold_set = None
    
        return samples

    def apply_transition(self, state: State, transition: Transition):
        """
        Applies transition to state.
        - State transitions logic.
        """
        t = transition.action
        dep = transition.dependency

        if t == self.LA:
            # Table 1: (sigma|i, j|beta, A) -> (sigma, j|beta, A U {(j,l,i)})
            # 1. Create arc from Buffer[0] to Stack[-1]
            head = state.B[0].id
            dependent = state.S[-1].id
            state.A.add((head, dep, dependent))
            # 2. Pop Stack
            state.S.pop()

        elif t == self.RA: 
            # Table 1: (sigma|i, j|beta, A) -> (sigma|i|j, beta, A U {(i,l,j)})
            # 1. Create arc from Stack[-1] to Buffer[0]
            head = state.S[-1].id
            dependent = state.B[0].id
            state.A.add((head, dep, dependent))
            # 2. Push Buffer[0] to Stack
            item = state.B.pop(0)
            state.S.append(item)

        elif t == self.REDUCE: 
            # Table 1: (sigma|i, beta, A) -> (sigma, beta, A)
            # 1. Pop Stack
            state.S.pop()

        else: # SHIFT
            # Table 1: (sigma, i|beta, A) -> (sigma|i, beta, A)
            # 1. Push Buffer[0] to Stack
            item = state.B.pop(0)
            state.S.append(item)
    

    def gold_arcs(self, sent: list['Token']) -> set:
        gold_arcs = set([])
        for token in sent[1:]:
            gold_arcs.add((token.head, token.dep, token.id))
        return gold_arcs


if __name__ == "__main__":


    print("**************************************************")
    print("*               Arc-eager function               *")
    print("**************************************************\n")

    print("Creating the initial state for the sentence: 'The cat is sleeping.' \n")

    tree = [
        Token(0, "ROOT", "ROOT", "_", "_", "_", "_", "_"),
        Token(1, "The", "the", "DET", "_", "Definite=Def|PronType=Art", 2, "det"),
        Token(2, "cat", "cat", "NOUN", "_", "Number=Sing", 4, "nsubj"),
        Token(3, "is", "be", "AUX", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", 4, "cop"),
        Token(4, "sleeping", "sleep", "VERB", "_", "VerbForm=Ger", 0, "root"),
        Token(5, ".", ".", "PUNCT", "_", "_", 4, "punct")
    ]

    arc_eager = ArcEager()
    print("Initial state")
    state = arc_eager.create_initial_state(tree)
    print(state)

    #Checking that is a final state
    print (f"Is the initial state a valid final state (buffer is empty)? {arc_eager.final_state(state)}\n")

    # Applying a SHIFT transition
    transition1 = Transition(arc_eager.SHIFT)
    arc_eager.apply_transition(state, transition1)
    print("State after applying the SHIFT transition:")
    print(state, "\n")

    #Obtaining the gold_arcs of the sentence with the function gold_arcs
    gold_arcs = arc_eager.gold_arcs(tree)
    print (f"Set of gold arcs: {gold_arcs}\n\n")


    print("**************************************************")
    print("*  Creating instances of the class Transition    *")
    print("**************************************************")

    # Creating a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)
    # Printing the created transition
    print(f"Created Transition: {shift_transition}")  # Output: Created Transition: SHIFT

    # Creating a LEFT-ARC transition with a specific dependency type
    left_arc_transition = Transition(ArcEager.LA, "nsubj")
    # Printing the created transition
    print(f"Created Transition: {left_arc_transition}")

    # Creating a RIGHT-ARC transition with a specific dependency type
    right_arc_transition = Transition(ArcEager.RA, "amod")
    # Printing the created transition
    print(f"Created Transition: {right_arc_transition}")

    # Creating a REDUCE transition
    reduce_transition = Transition(ArcEager.REDUCE)
    # Printing the created transition
    print(f"Created Transition: {reduce_transition}")  # Output: Created Transition: SHIFT

    print()
    print("**************************************************")
    print("*     Creating instances of the class  Sample    *")
    print("**************************************************")

    # For demonstration, let's create a dummy State instance
    state = arc_eager.create_initial_state(tree)  # Replace with actual state initialization as per your implementation

    # Create a Transition instance. For example, a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)

    # Now, create a Sample instance using the state and transition
    sample_instance = Sample(state, shift_transition)

    # To display the created Sample instance
    print("Sample:\n", sample_instance)