"""Tests for FigureIntegrityAntibodySystem — 6 figure caption verification antibodies."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from immunos_mcp.agents.figure_integrity_antibodies import (
    FigureIntegrityAntibodySystem,
    CaptionCompletenessAntibody,
    RepresentativeClaimAntibody,
    QuantificationAntibody,
    PanelConsistencyAntibody,
    DataPresentationAntibody,
    ImageSourceAntibody,
    create_figure_integrity_antibody_system,
)
from immunos_mcp.core.immune_response import ImmuneResponse


# -- System-Level Tests -------------------------------------------------------


class TestFigureIntegrityAntibodySystem:
    def test_create_system(self):
        system = FigureIntegrityAntibodySystem()
        assert len(system.antibodies) == 6
        assert set(system.antibodies.keys()) == {
            "caption_completeness", "representative_claim", "quantification",
            "panel_consistency", "data_presentation", "image_source",
        }

    def test_convenience_constructor(self):
        system = create_figure_integrity_antibody_system()
        assert isinstance(system, FigureIntegrityAntibodySystem)

    def test_training_status_initially_untrained(self):
        system = FigureIntegrityAntibodySystem()
        status = system.get_training_status()
        assert all(v is False for v in status.values())

    def test_good_caption_passes(self):
        system = FigureIntegrityAntibodySystem()
        caption = (
            "Figure 1. (A) Western blot analysis (n=3 biological replicates). "
            "(B) Quantification of panel A showing fold change. Error bars: SEM. "
            "*p<0.05 by Student's t-test. Scale bar: 50 um. Source data in Supplementary Table 1."
        )
        result = system.verify_figures(caption)
        assert result.total_checks == 6

    def test_empty_caption_flagged(self):
        system = FigureIntegrityAntibodySystem()
        result = system.verify_figures("")
        assert result.anomaly_count == 6

    def test_to_dict(self):
        system = FigureIntegrityAntibodySystem()
        result = system.verify_figures("Figure 1. Results (n=5, p<0.01, t-test).")
        d = result.to_dict()
        assert "is_suspicious" in d
        assert "response" in d
        assert d["response"] in ("ignore", "review", "reject")

    def test_response_field_exposed(self):
        system = FigureIntegrityAntibodySystem()
        result = system.verify_figures("Figure 1. Data shown.")
        assert isinstance(result.response, ImmuneResponse)

    def test_train_unknown_component_raises(self):
        system = FigureIntegrityAntibodySystem()
        with pytest.raises(ValueError, match="Unknown component"):
            system.train_antibody("resolution", ["test"])


# -- CaptionCompletenessAntibody -----------------------------------------------


class TestCaptionCompletenessAntibody:
    def test_complete_caption_passes(self):
        ab = CaptionCompletenessAntibody()
        result = ab.check("Western blot (n=3). Error bars: SEM. *p<0.05 by t-test. Scale bar: 50 um.")
        assert result.is_anomaly is False

    def test_missing_n_and_stats_flagged(self):
        ab = CaptionCompletenessAntibody()
        result = ab.check("Figure shows the results of the experiment.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = CaptionCompletenessAntibody()
        features = ab.extract_features("n=5, p<0.01, SEM, t-test")
        assert isinstance(features, np.ndarray)
        assert len(features) == 10


# -- RepresentativeClaimAntibody -----------------------------------------------


class TestRepresentativeClaimAntibody:
    def test_representative_with_n_passes(self):
        ab = RepresentativeClaimAntibody()
        result = ab.check("Representative images from n=5 independent experiments. Quantification in panel B.")
        assert result.is_anomaly is False

    def test_representative_without_n_flagged(self):
        ab = RepresentativeClaimAntibody()
        result = ab.check("Representative image of the results.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = RepresentativeClaimAntibody()
        features = ab.extract_features("Representative of 3 replicates")
        assert len(features) == 10


# -- QuantificationAntibody ---------------------------------------------------


class TestQuantificationAntibody:
    def test_quantified_result_passes(self):
        ab = QuantificationAntibody()
        result = ab.check("Significant increase (p=0.003, Cohen's d=0.8). Individual data points shown.")
        assert result.is_anomaly is False

    def test_significant_without_p_flagged(self):
        ab = QuantificationAntibody()
        result = ab.check("The difference was significant between the two groups.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = QuantificationAntibody()
        features = ab.extract_features("p=0.01, fold change = 2.5, 95% CI")
        assert len(features) == 10


# -- PanelConsistencyAntibody --------------------------------------------------


class TestPanelConsistencyAntibody:
    def test_consistent_panels_pass(self):
        ab = PanelConsistencyAntibody()
        result = ab.check("(A) Control. (B) Treatment. (C) Quantification of A and B. Same scale across panels.")
        assert result.is_anomaly is False

    def test_feature_extraction(self):
        ab = PanelConsistencyAntibody()
        features = ab.extract_features("Panel A shows control, panel B shows treatment.")
        assert len(features) == 10


# -- DataPresentationAntibody --------------------------------------------------


class TestDataPresentationAntibody:
    def test_good_plot_passes(self):
        ab = DataPresentationAntibody()
        result = ab.check("Violin plot with individual data points overlaid. Color-blind friendly palette.")
        assert result.is_anomaly is False

    def test_bar_without_points_flagged(self):
        ab = DataPresentationAntibody()
        result = ab.check("Bar chart showing mean values for each group.")
        assert result.is_anomaly is True

    def test_3d_chart_flagged(self):
        ab = DataPresentationAntibody()
        result = ab.check("3D bar chart showing the comparison between groups.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = DataPresentationAntibody()
        features = ab.extract_features("Box plot with jitter overlay")
        assert len(features) == 10


# -- ImageSourceAntibody -------------------------------------------------------


class TestImageSourceAntibody:
    def test_original_image_passes(self):
        ab = ImageSourceAntibody()
        result = ab.check("Original confocal images acquired on Zeiss LSM 880. Processing: ImageJ v1.53.")
        assert result.is_anomaly is False

    def test_stock_image_flagged(self):
        ab = ImageSourceAntibody()
        result = ab.check("Stock image from Shutterstock used for illustration.")
        assert result.is_anomaly is True

    def test_feature_extraction(self):
        ab = ImageSourceAntibody()
        features = ab.extract_features("Acquired with Nikon Eclipse microscope")
        assert len(features) == 10


# -- Discrimination Tests -----------------------------------------------------


class TestFigureIntegrityDiscrimination:
    """Test that the system discriminates between good and poor captions."""

    GOOD_CAPTIONS = [
        "Figure 1. (A) Western blot analysis (n=3 biological replicates). (B) Quantification of A. Error bars: SEM. *p<0.05 by t-test. Scale bar: 50 um. Source data in Supplementary Table 1. Images acquired on Bio-Rad ChemiDoc.",
        "Figure 2. Dose-response curve (n=6 per group). Individual data points shown. Mean +/- SD. One-way ANOVA with Tukey post-hoc. Violin plot shows full distribution. All panels use consistent axes.",
    ]

    POOR_CAPTIONS = [
        "Figure. Bar chart showing significant results.",
        "Picture of the experiment image showing the outcome between treatment versus control.",
    ]

    def test_good_captions_fewer_flags(self):
        system = FigureIntegrityAntibodySystem()
        for caption in self.GOOD_CAPTIONS:
            result = system.verify_figures(caption)
            assert result.anomaly_count <= 3, f"Too many flags: {caption[:60]}..."

    def test_poor_captions_more_flags(self):
        system = FigureIntegrityAntibodySystem()
        for caption in self.POOR_CAPTIONS:
            result = system.verify_figures(caption)
            assert result.anomaly_count >= 2, f"Too few flags: {caption[:60]}..."
