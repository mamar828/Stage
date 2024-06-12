class Coord:
    def __init__(self, degrees: float):
        self.degrees = degrees

    def __str__(self) -> str:
        return f"{self.__class__.__name__} : {self.degrees:.4f}°"



class RA(Coord):
    @classmethod
    def from_sexagesimal(cls, value: str):
        hours, minutes, seconds = [float(val) for val in value.split(":")]
        degrees = (hours*3600 + minutes*60 + seconds) / (24*3600) * 360
        return cls(degrees)
    
    def to_sexagesimal(self) -> str:
        total_seconds = self.degrees / 360 * (24*3600)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) / 3600 * 60)
        seconds = total_seconds - hours * 3600 - minutes * 60
        return f"{hours}:{minutes:02d}:{seconds:02.3f}"



class DEC(Coord):
    @classmethod
    def from_sexagesimal(cls, value: str):
        whole_degrees, minutes, seconds = [float(val) for val in value.split(":")]
        degrees = whole_degrees + minutes / 60 + seconds / 3600
        return cls(degrees)
    
    def to_sexagesimal(self) -> str:
        whole_degrees = int(self.degrees)
        minutes = int((self.degrees % 1) * 60)
        seconds = ((self.degrees % 1) * 60 - minutes) * 60
        return f"{whole_degrees}:{minutes:02d}:{seconds:02.3f}"




# ra = RA.from_sexagesimal("8:28:20.2402")
# print(ra.to_sexagesimal())
dec = DEC.from_sexagesimal("60:14:27.499")
print(dec)
print(dec.to_sexagesimal())

# 8:28:20.2402 -> 127.0843340
# 60:14:27.499 -> 60.2409721