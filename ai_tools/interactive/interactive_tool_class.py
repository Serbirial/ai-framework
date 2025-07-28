from abc import ABC, abstractmethod

class InteractiveTool(ABC):
    def __init__(self, tool_config=None, max_steps=20, verbose=False, streamer: object = None):
        """
        Base interactive tool framework.

        Args:
            max_steps (int): max interaction steps before forced stop.
            tool_config (dict): arbitrary config per tool.
            verbose (bool): enable debug logs.
            streamer (streamer object with add_special func): Streams special output to the streamer.
        """
        self.config = tool_config or {}
        
        self.streamer = streamer
        
        self.verbose = verbose
        self.max_steps = self.config.get("MAX_STEPS", max_steps)
        self.step_count = 0
        self.history = []  # list of (input, output) tuples or dicts
        self.done = False
        
        self.hook_start()


    def log(self, msg):
        if self.verbose:
            print(f"[InteractiveTool] {msg}")

    @abstractmethod
    def send_input(self, input_data):
        """
        Send input/command to the external system or tool.

        Args:
            input_data: string or structured command data.

        Returns:
            None
        """
        pass

    @abstractmethod
    def receive_output(self):
        """
        Receive output from the external system or tool.

        Returns:
            output_data: string or structured data
        """
        pass

    @abstractmethod
    def process_response(self, response):
        """
        Process a response from the tool and decide next action.

        Args:
            response: output_data received from the tool.

        Returns:
            continue_interaction (bool): True if interaction should continue.
        """
        pass

    @abstractmethod
    def describe(self) -> dict:
        """
        Return a dictionary describing the tool’s capabilities and state.
        Must be implemented in all subclasses.
        """
        pass


    def is_done(self):
        """
        Check if interaction is done or max steps exceeded.
        """
        return self.done or (self.step_count >= self.max_steps)

    def run_session(self, initial_input=None):
        """
        Main loop to run the interactive session.

        Args:
            initial_input: initial command or data to send.

        Returns:
            final_output: whatever the tool produces as final result.
        """
        if initial_input is not None:
            self.send_input(initial_input)

        while not self.is_done():
            self.step_count += 1
            self.log(f"Step {self.step_count} started.")

            response = self.receive_output()
            self.log(f"Received output: {response}")

            continue_interaction = self.process_response(response)
            if not continue_interaction:
                self.done = True
                break

            # prepare next input (by default, no-op)
            next_input = self.prepare_next_input()
            if next_input is not None:
                self.send_input(next_input)

        self.log(f"Interaction done after {self.step_count} steps.")
        return self.get_final_output()

    def prepare_next_input(self):
        """
        Hook to prepare the next input to send.
        Override in subclass if needed.

        Returns:
            next_input or None
        """
        return None

    def get_final_output(self):
        """
        Hook to collect and return final output.
        Override in subclass if needed.
        """
        return None

    def hook_start(self):
        """
        Hook that runs once on tool start/init.
        Override in subclasses for custom startup logic.
        """
        for method_name in getattr(self, "_hook_start_methods", []):
            method = getattr(self, method_name, None)
            if callable(method):
                method()