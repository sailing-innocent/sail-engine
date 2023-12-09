import pytest 
import collections
from random import choice

Card = collections.namedtuple('Card', ['rank', 'suit'])

# namedtuple was introduced into python since v2.6 for creating some classes with a few attributes but no methods
# like Database Index
class FrenchDeck:
    ranks = [str(n) for n in range(2,11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                                        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]

@pytest.mark.func 
def test_beer_card():
    beer_card = Card('7', 'diamonds')
    assert beer_card[0] == '7'
    assert beer_card[1] == 'diamonds'

@pytest.mark.func 
def test_french_deck():
    deck = FrenchDeck()
    assert len(deck) == 52
    assert (Card('Q','hearts') in deck) == True
