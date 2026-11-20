# Word length

`word_length` counts the characters of each region's text, punctuation
included. Characters are Unicode code points after NFC normalization, so an
accented letter counts once however the input encodes it. It has no config
keys. The output column is `word_length`.
