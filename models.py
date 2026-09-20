"""
Shared LSTM model definitions for crane forecasting.

Importing from a single module prevents the class definitions
from drifting out of sync between training and inference scripts.
"""

import torch
import torch.nn as nn


class CraneTiltLSTM(nn.Module):
    """
    Sequence-to-vector LSTM that forecasts structural tilt.

    Input:
        [batch, input_steps, num_features]

    Output:
        [batch, forecast_steps]  (normalised tilt values)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        forecast_steps,
        dropout
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, forecast_steps)
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        # Last timestep hidden state summarises history
        last_hidden = lstm_output[:, -1, :]

        return self.forecast_head(last_hidden)


class CraneRiskLSTM(nn.Module):
    """
    Sequence-to-vector LSTM that forecasts stability margin ratio.

    Input:
        [batch, input_steps, num_features]

    Output:
        [batch, forecast_steps]  (normalised margin_ratio values)

    Predicting margin_ratio directly (rather than tilt) is more
    useful for risk classification: the classifier threshold is
    defined in margin_ratio space.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        forecast_steps,
        dropout
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, forecast_steps)
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        last_hidden = lstm_output[:, -1, :]

        return self.forecast_head(last_hidden)
