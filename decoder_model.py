
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, c=64):
        super(Decoder, self).__init__()
        self.c = c

        self.deconv_stack = nn.Sequential(
            nn.ConvTranspose2d(c, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        return self.deconv_stack(x)

if __name__ == '__main__':
    # This is an example of how you would convert the model to ONNX format.
    # You would run this script on your server/development machine.

    # 1. Instantiate the model
    decoder = Decoder(c=64)
    decoder.eval()

    # 2. Create a dummy input with the correct shape (batch_size, channels, height, width)
    # The input shape must match the output shape of the encoder.
    # From server_encode.py, we know the latent shape is (w: 80, h: 45) for a 1280x720 input.
    # Let's calculate it dynamically for a 1280x720 input.
    def get_latent_shape(original_w, original_h):
        w, h = original_w, original_h
        for _ in range(4):
            w = (w - 5 + 2 * 2) // 2 + 1
            h = (h - 5 + 2 * 2) // 2 + 1
        return w, h

    latent_w, latent_h = get_latent_shape(1280, 720)
    dummy_input = torch.randn(1, 64, latent_h, latent_w)

    # 3. Define the output file name
    onnx_file = "decoder.onnx"

    # 4. Export the model
    try:
        torch.onnx.export(
            decoder,
            dummy_input,
            onnx_file,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Model successfully exported to {onnx_file}")
        print("Please place this file in the same directory as your client.html.")
    except Exception as e:
        print(f"Error exporting model: {e}")

