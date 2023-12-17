import random

import pytest
import torch

from danling.tensors import NestedTensor


class Test:
    def test_compare(self):
        value = 999999
        small = NestedTensor([[-value, -value, -value], [-value, -value]])
        big = abs(small)
        zero = 0
        assert (big > small).all()
        assert (big > small.tensor).all()
        assert (big > zero).all()
        assert (big > torch.tensor(zero)).all()
        assert (big >= small).all()
        assert (big >= small.tensor).all()
        assert (big >= zero).all()
        assert (big >= torch.tensor(zero)).all()
        assert (big == value).all()
        assert (big == big.tensor).all()
        assert (small < big).all()
        assert (small < big.tensor).all()
        assert (small < zero).all()
        assert (small < torch.tensor(zero)).all()
        assert (small <= big).all()
        assert (small <= big.tensor).all()
        assert (small <= zero).all()
        assert (small <= torch.tensor(zero)).all()
        with pytest.raises(TypeError):
            assert small < "small"
        with pytest.raises(TypeError):
            assert small > "small"
        with pytest.raises(TypeError):
            assert small <= "small"
        with pytest.raises(TypeError):
            assert small >= "small"
        assert small != "small"

    @pytest.mark.parametrize("test_input", [1, 0, -1, random.random()])
    def test_add_number(self, test_input):
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]])
        assert (nested_tensor + test_input == test_input + nested_tensor).all()

    @pytest.mark.parametrize("test_input", [torch.tensor([[6, 5, 4], [3, 2, 1]]), torch.randn(2, 3)])
    def test_add_tensor(self, test_input):
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]])
        assert (nested_tensor + test_input == nested_tensor.nested_like(test_input + nested_tensor)).all()

    @pytest.mark.parametrize("test_input", [NestedTensor([[5, 4, 3], [2, 1]])])
    def test_add_nested_tensor(self, test_input):
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]])
        assert (nested_tensor + test_input == test_input + nested_tensor).all()

    def test_iadd(self):
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]])
        nested_tensor += 1
        assert (nested_tensor == NestedTensor([[2, 3, 4], [5, 6]])).all()
        nested_tensor += torch.ones(2, 3)
        assert (nested_tensor == NestedTensor([[3, 4, 5], [6, 7]])).all()
        nested_tensor += NestedTensor([[5, 4, 3], [2, 1]])
        assert (nested_tensor == NestedTensor([[8, 8, 8], [8, 8]])).all()

    @pytest.mark.parametrize("test_input", [1, 0, -1, random.randint(0, 9)])
    def test_and_number(self, test_input):
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]]).int()
        assert (nested_tensor & test_input == test_input & nested_tensor).all()

    @pytest.mark.parametrize("test_input", [torch.tensor([[6, 5, 4], [3, 2, 1]]), torch.randint(0, 9, (2, 3))])
    def test_and_tensor(self, test_input):
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]]).int()
        assert (nested_tensor & test_input == nested_tensor.nested_like(test_input & nested_tensor)).all()

    @pytest.mark.parametrize("test_input", [NestedTensor([[5, 4, 3], [2, 1]])])
    def test_and_nested_tensor(self, test_input):
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]]).int()
        assert (nested_tensor & test_input == test_input & nested_tensor).all()

    def test_iand(self):
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]]).int()
        nested_tensor &= 1
        assert (nested_tensor == NestedTensor([[1, 0, 1], [0, 1]])).all()
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]]).int()
        nested_tensor &= torch.full((2, 3), 2)
        assert (nested_tensor == NestedTensor([[0, 2, 2], [0, 0]])).all()
        nested_tensor = NestedTensor([[1, 2, 3], [4, 5]]).int()
        nested_tensor &= NestedTensor([[5, 4, 3], [2, 1]])
        assert (nested_tensor == NestedTensor([[1, 0, 3], [0, 1]])).all()
