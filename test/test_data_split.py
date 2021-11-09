from modist_import import modist
from modist.data_split import split_mp3
from icecream import ic


def test_split_mp3():
    splits = split_mp3()
    ic(splits)
    assert len(splits)==2
    assert len(splits[1]) == 16    

    splits_2 = split_mp3()
    test_2 = splits_2[1]
    test_1 = splits[1]
    assert all(x==y for x, y in zip(test_1, test_2))    