import torch
from cepler.models.transformer import CustomTransformer

def main():
    # Определяем устройство автоматически
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CustomTransformer(
        vocab_size=5000,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        num_classes=2,
        max_len=128,
        attention_type='ray',
        num_rays=8,
        sparse_rays=True,
        residual_echo=True,
    ).to(device)

    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 5000, (batch_size, seq_len)).to(device)

    # Обычный проход
    logits, _, _ = model(input_ids)
    print(f"Logits shape: {logits.shape}")

    # Ранний выход
    logits, exit_block, confidence = model(input_ids, exit_threshold=0.95)
    print(f"Exit block: {exit_block}, confidence: {confidence}")

if __name__ == "__main__":
    main()