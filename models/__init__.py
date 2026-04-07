"""HealthBeauty360 ML models package."""

from models.churn_prediction import ChurnPredictionModel
from models.customer_segmentation import CustomerSegmentationModel
from models.demand_forecast import DemandForecastModel
from models.inventory_scoring import InventoryScoringModel
from models.trend_detection import TrendDetectionModel

__all__ = [
	"ChurnPredictionModel",
	"CustomerSegmentationModel",
	"DemandForecastModel",
	"InventoryScoringModel",
	"TrendDetectionModel",
]
