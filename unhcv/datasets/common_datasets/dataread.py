class BaseDataRead:
    def __init__(self) -> None:
        pass

    def read_next(self):
        raise NotImplementedError
    
    def read_with_key(self):
        raise NotImplementedError
    
    def read_i(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
    