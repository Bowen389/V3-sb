import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import scan


class RecoveryTests(unittest.TestCase):
    def bars(self, end='2026-11-18'):
        dates = pd.bdate_range('2026-07-01', end)
        return pd.DataFrame({'日期': dates, '开盘': 10., '收盘': 10.,
                             '最高': 11., '最低': 9., '成交额': 100_000_000.})

    def test_stale_nonempty_primary_uses_backup_and_cache(self):
        with tempfile.TemporaryDirectory() as root, patch('scan.Path', side_effect=lambda p: Path(root) / p):
            with patch.object(scan.ak, 'stock_zh_a_hist_em', return_value=self.bars('2026-11-17')), \
                 patch.object(scan.ak, 'stock_zh_a_hist_sina', return_value=self.bars()) as backup:
                result = scan.history('002685', '', '20260701', '20261118', pd.Timestamp('2026-11-18'))
                self.assertEqual(result.iloc[-1]['日期'], pd.Timestamp('2026-11-18'))
                backup.assert_called_once()
            with patch.object(scan.ak, 'stock_zh_a_hist_em', side_effect=AssertionError('cache should suffice')):
                self.assertEqual(len(scan.history('002685', '', '20260701', '20261118', pd.Timestamp('2026-11-18'))), len(result))

    def test_missing_amount_is_not_silently_replaced_with_volume(self):
        bad = self.bars().drop(columns='成交额')
        with tempfile.TemporaryDirectory() as root, patch('scan.Path', side_effect=lambda p: Path(root) / p), \
             patch.object(scan.ak, 'stock_zh_a_hist_em', return_value=bad), \
             patch.object(scan.ak, 'stock_zh_a_hist_sina', return_value=bad), \
             patch.object(scan.ak, 'stock_zh_a_hist_tx', return_value=bad):
            with self.assertRaisesRegex(RuntimeError, '字段不完整'):
                scan.history('002685', '', '20260701', '20261118', pd.Timestamp('2026-11-18'))

    def test_tencent_raw_fallback_requires_real_amount(self):
        with tempfile.TemporaryDirectory() as root, patch('scan.Path', side_effect=lambda p: Path(root) / p), \
             patch.object(scan.ak, 'stock_zh_a_hist_em', side_effect=RuntimeError('rate limited')), \
             patch.object(scan.ak, 'stock_zh_a_hist_sina', side_effect=RuntimeError('timeout')), \
             patch.object(scan.ak, 'stock_zh_a_hist_tx', return_value=self.bars()) as backup:
            result = scan.history('002685', '', '20260701', '20261118', pd.Timestamp('2026-11-18'))
            self.assertEqual(result.iloc[-1]['成交额'], 100_000_000.)
            backup.assert_called_once()

    def run_partial(self, preselect):
        pool = pd.DataFrame({'代码': ['002685', '600001'], '名称': ['缺价票', '正常票'], '连板数': [1, 1]})
        def history(code, *args):
            if code == '002685':
                raise RuntimeError('missing bar')
            return self.bars()
        argv = ['scan.py', '--buy-date', '2026-11-18'] + (['--preselect'] if preselect else [])
        with tempfile.TemporaryDirectory() as root, patch('scan.Path', side_effect=lambda p: Path(root) / p), \
             patch('sys.argv', argv), patch.object(scan.ak, 'stock_zt_pool_em', return_value=pool), \
             patch('scan.history', side_effect=history), patch('scan.time.sleep'):
            scan.main()
            stage = 'preselect' if preselect else 'simulation'
            quality = json.loads((Path(root) / f'reports/quality/2026-11-18-{stage}.json').read_text())
            self.assertEqual(quality['status'], 'partial')
            self.assertEqual(quality['resolved_count'], 1)
            self.assertIn('002685', quality['errors'])
            path = 'reports/preselect/2026-11-18.md' if preselect else 'reports/2026-11-18.md'
            report = (Path(root) / path).read_text()
            self.assertIn('部分候选' if preselect else '暂不生成最终买入名单', report)
            if preselect:
                self.assertIn('600001', report)

    def test_one_stock_failure_preserves_partial_candidates(self):
        self.run_partial(True)

    def test_incomplete_pool_blocks_final_ranking(self):
        self.run_partial(False)


if __name__ == '__main__':
    unittest.main()
