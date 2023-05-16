import os


class ConfigParser:

    def __init__(self):
        self._directory = os.path.dirname(os.path.abspath(__file__))
        self._heroes = self._load_heroes()
        self._styles = self._load_styles()
        self._hero2index = {hero: i for i, hero in enumerate(self._heroes)}

    def _load_heroes(self):
        heroes = []
        with open(os.path.join(self._directory, "heroes.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                heroes.append(line)
        return heroes

    def _load_styles(self):
        styles = []
        with open(os.path.join(self._directory, "styles.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                styles.append(line)
        return styles

    @property
    def heroes(self):
        return self._heroes

    @property
    def styles(self):
        return self._styles

    @property
    def hero2index(self):
        return self._hero2index
