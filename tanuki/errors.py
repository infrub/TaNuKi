class UndecidedError(Exception):
    pass
        
class LengthError(Exception):
    pass

class LabelsLengthError(LengthError):
    pass

class LabelsTypeError(TypeError):
    pass

class ShapeError(ValueError):
    pass

class IndicesLengthError(LengthError):
    pass

class SitesLengthError(LengthError):
    pass

class CantKeepDiagonalityError(Exception):
    pass

class ArgumentError(Exception):
    pass

class InternalError(Exception):
    pass