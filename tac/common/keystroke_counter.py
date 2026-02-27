from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock

class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda:0)
        self.key_press_list = list()
        self.lock = Lock()
        super().__init__(on_press=self._on_press, on_release=self._on_release)
    
    def _on_press(self, key):
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)
    
    def _on_release(self, key):
        pass
    
    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda:0)
            self.key_press_list = list()
    
    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]
    
    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list.clear()
            return events

if __name__ == '__main__':
    import time
    print("Starting KeystrokeCounter test. Press keys to see events. Press Ctrl+C to exit.")
    with KeystrokeCounter() as counter:
        try:
            while True:
                events = counter.get_press_events()
                if events:
                    print("Pressed keys since last check:", events)
                
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting test.")
