
import torch
import torch.nn as nn


class CompressorLSTM(nn.Module):

    """
    LSTM model for industrial compressor forecasting.
    """

    def __init__(
        self,
        input_size=8,
        hidden_size=64,
        num_layers=2,
        output_size=6,
        dropout=0.2
    ):

        super(CompressorLSTM, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout
        )

        # Fully connected output layer
        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        """
        Forward pass.

        Input shape:
            (batch_size, sequence_length, input_size)

        Output shape:
            (batch_size, output_size)
        """

        # Pass input through LSTM
        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        # Select the last time step
        last_output = lstm_output[:, -1, :]

        # Generate sensor predictions
        output = self.fc(last_output)

        return output


if __name__ == "__main__":

    print("\n========== LSTM MODEL TEST ==========")

    # Initialize model
    model = CompressorLSTM()

    # Display architecture
    print("\nModel Architecture:")

    print(model)

    # Create sample input
    sample_input = torch.randn(
        32,
        30,
        8
    )

    # Forward pass
    predictions = model(sample_input)

    print("\nInput shape:", sample_input.shape)

    print("Output shape:", predictions.shape)

    print("\nTotal trainable parameters:")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(total_parameters)

    print("\nLSTM model created successfully!")
