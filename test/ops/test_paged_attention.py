import unittest
import torch
from unittest.mock import patch, MagicMock
import numpy as np
from ops.paged_attention import paged_attention


class TestPagedAttention(unittest.TestCase):
    def setUp(self):
        self.mock_kernel = MagicMock()
        self.patcher = patch(
            '__main__.paged_attention_kernel',
            new=self.mock_kernel
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_basic_functionality(self):
        """Test basic functionality with single sequence"""
        query = torch.randn(1, 2, 64)  # seq_len=1, head_num=2, head_size=64
        key_cache = torch.randn(3, 2, 64, 32)  # block_num=3, block_size=32
        value_cache = torch.randn(3, 2, 64, 32)
        block_table = [[0, 1]]  # single sequence with 2 blocks
        
        result = paged_attention(query, key_cache, value_cache, block_table)
        
        self.mock_kernel.assert_called_once()
        self.assertEqual(result.shape, query.shape)
        self.assertTrue(torch.equal(result, self.mock_kernel.return_value))

    def test_multi_sequence(self):
        """Test with multiple sequences"""
        query = torch.randn(3, 4, 128)  # 3 sequences
        key_cache = torch.randn(5, 4, 128, 64)  # 5 blocks
        value_cache = torch.randn(5, 4, 128, 64)
        block_table = [[0], [1, 2], [3, 4]]  # 3 sequences
        
        result = paged_attention(query, key_cache, value_cache, block_table)
        
        self.assertEqual(self.mock_kernel.call_count, 3)
        self.assertEqual(result.shape, query.shape)

    def test_empty_sequence(self):
        """Test with empty sequence (should still work)"""
        query = torch.randn(1, 1, 16)
        key_cache = torch.randn(2, 1, 16, 8)
        value_cache = torch.randn(2, 1, 16, 8)
        block_table = [[]]  # empty sequence
        
        result = paged_attention(query, key_cache, value_cache, block_table)
        
        self.mock_kernel.assert_called_once()
        self.assertEqual(result.shape, query.shape)

    def test_shape_mismatch(self):
        """Test when shapes don't match"""
        query = torch.randn(1, 2, 64)
        key_cache = torch.randn(3, 4, 64, 32)  # head_num mismatch
        value_cache = torch.randn(3, 2, 64, 32)
        block_table = [[0]]
        
        with self.assertRaises(RuntimeError):
            paged_attention(query, key_cache, value_cache, block_table)

    def test_invalid_block_table(self):
        """Test with invalid block table references"""
        query = torch.randn(1, 1, 8)
        key_cache = torch.randn(1, 1, 8, 4)
        value_cache = torch.randn(1, 1, 8, 4)
        block_table = [[1]]  # invalid block index
        
        with self.assertRaises(IndexError):
            paged_attention(query, key_cache, value_cache, block_table)

if __name__ == '__main__':
    unittest.main()