"""
Unit tests for metrics and evaluation utilities.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.drug_combo.utils.metrics import (
    calculate_metrics,
    calculate_interaction_metrics,
    precision_score,
    plot_prediction_scatter,
    plot_gene_wise_performance,
    plot_interaction_analysis,
    create_evaluation_report
)


class TestCalculateMetrics:
    """Test cases for calculate_metrics function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample prediction and target data."""
        np.random.seed(42)
        n_samples, n_genes = 100, 50
        
        # Create realistic predictions and targets
        targets = np.random.randn(n_samples, n_genes)
        
        # Create predictions with some correlation to targets
        predictions = targets + np.random.randn(n_samples, n_genes) * 0.5
        
        return predictions, targets
    
    def test_calculate_metrics_basic(self, sample_data):
        """Test basic metrics calculation."""
        predictions, targets = sample_data
        
        metrics = calculate_metrics(predictions, targets)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'mae', 'mse', 'rmse', 'r2_overall',
            'pearson_corr', 'pearson_pvalue', 'spearman_corr', 'spearman_pvalue',
            'r2_gene_mean', 'r2_gene_std', 'r2_gene_median',
            'pearson_gene_mean', 'pearson_gene_std', 'pearson_gene_median',
            'genes_r2_positive', 'genes_r2_good', 'genes_pearson_high'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check that metrics are reasonable
        assert metrics['mae'] > 0
        assert metrics['mse'] > 0
        assert metrics['rmse'] > 0
        assert -1 <= metrics['r2_overall'] <= 1
        assert -1 <= metrics['pearson_corr'] <= 1
        assert -1 <= metrics['spearman_corr'] <= 1
        assert 0 <= metrics['pearson_pvalue'] <= 1
        assert 0 <= metrics['spearman_pvalue'] <= 1
    
    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        np.random.seed(42)
        targets = np.random.randn(50, 20)
        predictions = targets.copy()  # Perfect prediction
        
        metrics = calculate_metrics(predictions, targets)
        
        # Perfect prediction should give excellent metrics
        assert metrics['mae'] < 1e-10
        assert metrics['mse'] < 1e-10
        assert metrics['rmse'] < 1e-10
        assert metrics['r2_overall'] > 0.99
        assert metrics['pearson_corr'] > 0.99
        assert metrics['spearman_corr'] > 0.99
    
    def test_calculate_metrics_random_prediction(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        targets = np.random.randn(100, 30)
        predictions = np.random.randn(100, 30)  # Random prediction
        
        metrics = calculate_metrics(predictions, targets)
        
        # Random prediction should give poor metrics
        assert metrics['mae'] > 0.5
        assert metrics['mse'] > 0.5
        assert metrics['r2_overall'] < 0.5
        assert abs(metrics['pearson_corr']) < 0.5
    
    def test_calculate_metrics_gene_wise(self, sample_data):
        """Test gene-wise metrics calculation."""
        predictions, targets = sample_data
        
        metrics = calculate_metrics(predictions, targets)
        
        # Check gene-wise metrics
        assert 0 <= metrics['genes_r2_positive'] <= targets.shape[1]
        assert 0 <= metrics['genes_r2_good'] <= targets.shape[1]
        assert 0 <= metrics['genes_pearson_high'] <= targets.shape[1]
        
        # Mean should be between min and max
        assert metrics['r2_gene_mean'] >= -1  # R2 can be negative
        assert metrics['pearson_gene_mean'] >= -1
        assert metrics['pearson_gene_mean'] <= 1
        
        # Standard deviation should be non-negative
        assert metrics['r2_gene_std'] >= 0
        assert metrics['pearson_gene_std'] >= 0
    
    def test_calculate_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        # Test with constant predictions
        targets = np.random.randn(50, 10)
        predictions = np.ones_like(targets) * 0.5
        
        metrics = calculate_metrics(predictions, targets)
        
        # Should handle constant predictions gracefully
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['mse'])
        assert not np.isnan(metrics['r2_overall'])
        
        # Test with constant targets
        targets = np.ones((50, 10)) * 2.0
        predictions = np.random.randn(50, 10)
        
        metrics = calculate_metrics(predictions, targets)
        
        # Should handle constant targets gracefully
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['mse'])
    
    def test_calculate_metrics_single_gene(self):
        """Test metrics with single gene."""
        np.random.seed(42)
        targets = np.random.randn(100, 1)
        predictions = targets + np.random.randn(100, 1) * 0.3
        
        metrics = calculate_metrics(predictions, targets)
        
        # Should work with single gene
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['pearson_corr'])
        assert metrics['genes_r2_positive'] <= 1
        assert metrics['genes_r2_good'] <= 1
        assert metrics['genes_pearson_high'] <= 1


class TestCalculateInteractionMetrics:
    """Test cases for calculate_interaction_metrics function."""
    
    @pytest.fixture
    def interaction_data(self):
        """Create sample interaction data."""
        np.random.seed(42)
        n_samples, n_genes = 100, 30
        
        # Create single drug effects
        single_drug_a = np.random.randn(n_samples, n_genes) * 0.5
        single_drug_b = np.random.randn(n_samples, n_genes) * 0.5
        
        # Create combination effects (additive + interaction)
        additive = single_drug_a + single_drug_b
        interaction = np.random.randn(n_samples, n_genes) * 0.2
        targets = additive + interaction
        
        # Create predictions
        predictions = targets + np.random.randn(n_samples, n_genes) * 0.3
        
        return predictions, targets, single_drug_a, single_drug_b
    
    def test_calculate_interaction_metrics_basic(self, interaction_data):
        """Test basic interaction metrics calculation."""
        predictions, targets, single_drug_a, single_drug_b = interaction_data
        
        metrics = calculate_interaction_metrics(
            predictions, targets, single_drug_a, single_drug_b
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'additive_baseline_mae', 'model_mae', 'improvement_over_additive_pct',
            'interaction_mae', 'interaction_r2', 'interaction_classification_accuracy',
            'synergy_precision', 'antagonism_precision', 'additive_precision'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check that metrics are reasonable
        assert metrics['additive_baseline_mae'] > 0
        assert metrics['model_mae'] > 0
        assert metrics['interaction_mae'] > 0
        assert -1 <= metrics['interaction_r2'] <= 1
        assert 0 <= metrics['interaction_classification_accuracy'] <= 1
        assert 0 <= metrics['synergy_precision'] <= 1
        assert 0 <= metrics['antagonism_precision'] <= 1
        assert 0 <= metrics['additive_precision'] <= 1
    
    def test_calculate_interaction_metrics_perfect_additive(self):
        """Test metrics when predictions are perfectly additive."""
        np.random.seed(42)
        n_samples, n_genes = 50, 20
        
        single_drug_a = np.random.randn(n_samples, n_genes) * 0.5
        single_drug_b = np.random.randn(n_samples, n_genes) * 0.5
        targets = single_drug_a + single_drug_b  # Perfect additive
        predictions = targets.copy()  # Perfect prediction
        
        metrics = calculate_interaction_metrics(
            predictions, targets, single_drug_a, single_drug_b
        )
        
        # Perfect additive prediction should give excellent metrics
        assert metrics['model_mae'] < 1e-10
        assert metrics['interaction_mae'] < 1e-10
        assert metrics['improvement_over_additive_pct'] > 99
    
    def test_calculate_interaction_metrics_synergy(self):
        """Test metrics with synergistic interactions."""
        np.random.seed(42)
        n_samples, n_genes = 50, 20
        
        single_drug_a = np.random.randn(n_samples, n_genes) * 0.3
        single_drug_b = np.random.randn(n_samples, n_genes) * 0.3
        
        # Create strong synergistic effects
        additive = single_drug_a + single_drug_b
        synergy = np.abs(np.random.randn(n_samples, n_genes)) * 0.5  # Positive interactions
        targets = additive + synergy
        predictions = targets + np.random.randn(n_samples, n_genes) * 0.1
        
        metrics = calculate_interaction_metrics(
            predictions, targets, single_drug_a, single_drug_b
        )
        
        # Should detect synergistic interactions
        assert metrics['interaction_classification_accuracy'] > 0
        assert metrics['synergy_precision'] >= 0
    
    def test_calculate_interaction_metrics_antagonism(self):
        """Test metrics with antagonistic interactions."""
        np.random.seed(42)
        n_samples, n_genes = 50, 20
        
        single_drug_a = np.random.randn(n_samples, n_genes) * 0.3
        single_drug_b = np.random.randn(n_samples, n_genes) * 0.3
        
        # Create antagonistic effects
        additive = single_drug_a + single_drug_b
        antagonism = -np.abs(np.random.randn(n_samples, n_genes)) * 0.5  # Negative interactions
        targets = additive + antagonism
        predictions = targets + np.random.randn(n_samples, n_genes) * 0.1
        
        metrics = calculate_interaction_metrics(
            predictions, targets, single_drug_a, single_drug_b
        )
        
        # Should detect antagonistic interactions
        assert metrics['interaction_classification_accuracy'] > 0
        assert metrics['antagonism_precision'] >= 0


class TestPrecisionScore:
    """Test cases for precision_score function."""
    
    def test_precision_score_perfect(self):
        """Test precision score with perfect predictions."""
        y_true = np.array([True, True, False, False, True])
        y_pred = np.array([True, True, False, False, True])
        
        precision = precision_score(y_true, y_pred)
        assert precision == 1.0
    
    def test_precision_score_zero_precision(self):
        """Test precision score with zero precision."""
        y_true = np.array([False, False, False, False, False])
        y_pred = np.array([True, True, True, True, True])
        
        precision = precision_score(y_true, y_pred)
        assert precision == 0.0
    
    def test_precision_score_no_predictions(self):
        """Test precision score with no positive predictions."""
        y_true = np.array([True, True, False, False, True])
        y_pred = np.array([False, False, False, False, False])
        
        precision = precision_score(y_true, y_pred)
        assert precision == 0.0
    
    def test_precision_score_partial(self):
        """Test precision score with partial accuracy."""
        y_true = np.array([True, True, False, False, True])
        y_pred = np.array([True, False, False, True, True])
        
        precision = precision_score(y_true, y_pred)
        
        # TP = 2, FP = 1, so precision = 2/3
        assert abs(precision - 2/3) < 1e-10


class TestPlottingFunctions:
    """Test cases for plotting functions."""
    
    @pytest.fixture
    def sample_plot_data(self):
        """Create sample data for plotting."""
        np.random.seed(42)
        n_samples, n_genes = 100, 20
        
        targets = np.random.randn(n_samples, n_genes)
        predictions = targets + np.random.randn(n_samples, n_genes) * 0.5
        
        return predictions, targets
    
    def test_plot_prediction_scatter(self, sample_plot_data):
        """Test prediction scatter plot."""
        predictions, targets = sample_plot_data
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            with patch('matplotlib.pyplot.scatter') as mock_scatter:
                with patch('matplotlib.pyplot.plot') as mock_plot:
                    with patch('matplotlib.pyplot.show') as mock_show:
                        plot_prediction_scatter(predictions, targets)
                        
                        # Check that plotting functions were called
                        mock_figure.assert_called_once()
                        mock_scatter.assert_called_once()
                        mock_plot.assert_called_once()
                        mock_show.assert_called_once()
    
    def test_plot_prediction_scatter_with_save(self, sample_plot_data):
        """Test prediction scatter plot with saving."""
        predictions, targets = sample_plot_data
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            with patch('matplotlib.pyplot.scatter') as mock_scatter:
                with patch('matplotlib.pyplot.savefig') as mock_savefig:
                    with patch('matplotlib.pyplot.show') as mock_show:
                        plot_prediction_scatter(predictions, targets, save_path="test.png")
                        
                        # Check that save function was called
                        mock_savefig.assert_called_once_with("test.png", dpi=300, bbox_inches='tight')
    
    def test_plot_gene_wise_performance(self, sample_plot_data):
        """Test gene-wise performance plot."""
        predictions, targets = sample_plot_data
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            with patch('matplotlib.pyplot.show') as mock_show:
                plot_gene_wise_performance(predictions, targets)
                
                # Check that plotting functions were called
                mock_subplots.assert_called_once()
                mock_show.assert_called_once()
    
    def test_plot_interaction_analysis(self, sample_plot_data):
        """Test interaction analysis plot."""
        predictions, targets = sample_plot_data
        
        # Create single drug data
        single_drug_a = np.random.randn(*targets.shape) * 0.5
        single_drug_b = np.random.randn(*targets.shape) * 0.5
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            with patch('matplotlib.pyplot.show') as mock_show:
                plot_interaction_analysis(predictions, targets, single_drug_a, single_drug_b)
                
                # Check that plotting functions were called
                mock_subplots.assert_called_once()
                mock_show.assert_called_once()
    
    def test_plot_functions_with_edge_cases(self):
        """Test plotting functions with edge cases."""
        # Test with small data
        predictions = np.random.randn(5, 3)
        targets = np.random.randn(5, 3)
        
        with patch('matplotlib.pyplot.figure'):
            with patch('matplotlib.pyplot.scatter'):
                with patch('matplotlib.pyplot.show'):
                    plot_prediction_scatter(predictions, targets)
        
        # Test with single sample
        predictions = np.random.randn(1, 10)
        targets = np.random.randn(1, 10)
        
        with patch('matplotlib.pyplot.subplots'):
            with patch('matplotlib.pyplot.show'):
                plot_gene_wise_performance(predictions, targets)


class TestCreateEvaluationReport:
    """Test cases for create_evaluation_report function."""
    
    @pytest.fixture
    def evaluation_data(self):
        """Create sample evaluation data."""
        np.random.seed(42)
        n_samples, n_genes = 100, 30
        
        targets = np.random.randn(n_samples, n_genes)
        predictions = targets + np.random.randn(n_samples, n_genes) * 0.5
        
        single_drug_a = np.random.randn(n_samples, n_genes) * 0.3
        single_drug_b = np.random.randn(n_samples, n_genes) * 0.3
        
        return predictions, targets, single_drug_a, single_drug_b
    
    def test_create_evaluation_report_basic(self, evaluation_data):
        """Test basic evaluation report creation."""
        predictions, targets, single_drug_a, single_drug_b = evaluation_data
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch('matplotlib.pyplot.figure'):
                with patch('matplotlib.pyplot.subplots'):
                    with patch('matplotlib.pyplot.savefig'):
                        with patch('matplotlib.pyplot.show'):
                            with patch('builtins.open', create=True) as mock_open:
                                with patch('json.dump') as mock_json_dump:
                                    metrics = create_evaluation_report(
                                        predictions, targets, single_drug_a, single_drug_b
                                    )
                                    
                                    # Check that report was created
                                    assert isinstance(metrics, dict)
                                    assert 'mae' in metrics
                                    assert 'interaction_mae' in metrics
                                    
                                    # Check that directory was created
                                    mock_mkdir.assert_called_once()
                                    
                                    # Check that JSON was saved
                                    mock_json_dump.assert_called_once()
    
    def test_create_evaluation_report_without_interaction(self, evaluation_data):
        """Test evaluation report without interaction data."""
        predictions, targets, _, _ = evaluation_data
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch('matplotlib.pyplot.figure'):
                with patch('matplotlib.pyplot.subplots'):
                    with patch('matplotlib.pyplot.savefig'):
                        with patch('matplotlib.pyplot.show'):
                            with patch('builtins.open', create=True) as mock_open:
                                with patch('json.dump') as mock_json_dump:
                                    metrics = create_evaluation_report(predictions, targets)
                                    
                                    # Check that basic metrics were calculated
                                    assert isinstance(metrics, dict)
                                    assert 'mae' in metrics
                                    assert 'interaction_mae' not in metrics
    
    def test_create_evaluation_report_custom_directory(self, evaluation_data):
        """Test evaluation report with custom directory."""
        predictions, targets, single_drug_a, single_drug_b = evaluation_data
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch('matplotlib.pyplot.figure'):
                with patch('matplotlib.pyplot.subplots'):
                    with patch('matplotlib.pyplot.savefig'):
                        with patch('matplotlib.pyplot.show'):
                            with patch('builtins.open', create=True) as mock_open:
                                with patch('json.dump') as mock_json_dump:
                                    create_evaluation_report(
                                        predictions, targets, single_drug_a, single_drug_b,
                                        save_dir="custom_results"
                                    )
                                    
                                    # Check that custom directory was used
                                    mock_mkdir.assert_called_once()


class TestMetricsIntegration:
    """Integration tests for metrics module."""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        np.random.seed(42)
        n_samples, n_genes = 50, 20
        
        # Create realistic data
        targets = np.random.randn(n_samples, n_genes)
        predictions = targets + np.random.randn(n_samples, n_genes) * 0.3
        
        single_drug_a = np.random.randn(n_samples, n_genes) * 0.3
        single_drug_b = np.random.randn(n_samples, n_genes) * 0.3
        
        try:
            # Test basic metrics
            basic_metrics = calculate_metrics(predictions, targets)
            assert isinstance(basic_metrics, dict)
            assert len(basic_metrics) > 10
            
            # Test interaction metrics
            interaction_metrics = calculate_interaction_metrics(
                predictions, targets, single_drug_a, single_drug_b
            )
            assert isinstance(interaction_metrics, dict)
            assert len(interaction_metrics) > 5
            
            # Test plotting (mocked)
            with patch('matplotlib.pyplot.figure'):
                with patch('matplotlib.pyplot.scatter'):
                    with patch('matplotlib.pyplot.show'):
                        plot_prediction_scatter(predictions, targets)
            
            # Test evaluation report (mocked)
            with patch('pathlib.Path.mkdir'):
                with patch('matplotlib.pyplot.figure'):
                    with patch('matplotlib.pyplot.subplots'):
                        with patch('matplotlib.pyplot.savefig'):
                            with patch('matplotlib.pyplot.show'):
                                with patch('builtins.open', create=True):
                                    with patch('json.dump'):
                                        all_metrics = create_evaluation_report(
                                            predictions, targets, single_drug_a, single_drug_b
                                        )
                                        
                                        assert isinstance(all_metrics, dict)
                                        assert len(all_metrics) > 15
            
        except Exception as e:
            pytest.fail(f"Full evaluation pipeline failed: {e}")
    
    def test_metrics_with_different_data_sizes(self):
        """Test metrics with different data sizes."""
        np.random.seed(42)
        
        # Test with different sizes
        sizes = [(10, 5), (50, 20), (100, 50)]
        
        for n_samples, n_genes in sizes:
            targets = np.random.randn(n_samples, n_genes)
            predictions = targets + np.random.randn(n_samples, n_genes) * 0.5
            
            try:
                metrics = calculate_metrics(predictions, targets)
                
                # Check that metrics are reasonable
                assert isinstance(metrics, dict)
                assert not np.isnan(metrics['mae'])
                assert not np.isnan(metrics['mse'])
                assert not np.isnan(metrics['r2_overall'])
                assert not np.isnan(metrics['pearson_corr'])
                
            except Exception as e:
                pytest.fail(f"Metrics calculation failed for size {n_samples}x{n_genes}: {e}")
    
    def test_metrics_numerical_stability(self):
        """Test metrics with challenging numerical cases."""
        np.random.seed(42)
        
        # Test with very small values
        targets = np.random.randn(50, 10) * 1e-6
        predictions = targets + np.random.randn(50, 10) * 1e-7
        
        metrics = calculate_metrics(predictions, targets)
        
        # Should handle small values gracefully
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['mse'])
        assert not np.isnan(metrics['r2_overall'])
        
        # Test with large values
        targets = np.random.randn(50, 10) * 1e6
        predictions = targets + np.random.randn(50, 10) * 1e5
        
        metrics = calculate_metrics(predictions, targets)
        
        # Should handle large values gracefully
        assert not np.isnan(metrics['mae'])
        assert not np.isnan(metrics['mse'])
        assert not np.isnan(metrics['r2_overall'])


if __name__ == "__main__":
    pytest.main([__file__])