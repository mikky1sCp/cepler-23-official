import torch
from cepler.models.transformer import CustomTransformer

def main():
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
    ).cuda()

    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 5000, (batch_size, seq_len)).cuda()

    # Обычный проход
    logits, _, _ = model(input_ids)
    print(f"Logits shape: {logits.shape}")

    # Ранний выход
    logits, exit_block, confidence = model(input_ids, exit_threshold=0.95)
    print(f"Exit block: {exit_block}, confidence: {confidence}")

if __name__ == "__main__":
    main()