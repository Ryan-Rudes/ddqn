class Schedule:
    def __setattr__(self, name, value):
        # This does nothing at the moment because no implementations of
        # Schedule have attributes  marked with a leading underscore
        if name.startswith('_'):
            raise AttributeError(f"Attempted to set value of private attribute {name}, but Schedule object is read-only")
        object.__setattr__(self, name, value)

    def update(self):
        raise NotImplementedError
