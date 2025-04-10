import matplotlib.pyplot as plt
import os


def visualize_dataloader(dataloader, num_pairs=4, title="Sample Pairs", save_path=None):
    """
    Hiển thị một số cặp ảnh từ dataloader.

    Args:
        dataloader: Dataloader của tập train hoặc valid.
        num_pairs: Số lượng cặp ảnh muốn hiển thị.
        title: Tiêu đề chính cho biểu đồ.
        save_path: Đường dẫn để lưu ảnh (nếu có).
    """
    shown = 0
    fig, axes = plt.subplots(num_pairs, 2, figsize=(4, num_pairs * 2))

    if num_pairs == 1:
        axes = [axes]  # Nếu chỉ có 1 cặp thì axes là một cặp subplot (không phải list of lists)

    for (img1_batch, img2_batch), labels in dataloader:
        batch_size = img1_batch.size(0)
        for i in range(batch_size):
            if shown >= num_pairs:
                fig.suptitle(title, fontsize=14)
                plt.tight_layout()

                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    print(f"✅ Saved to {save_path}")
                    plt.close()
                else:
                    plt.show()
                return

            img1 = img1_batch[i].permute(1, 2, 0).cpu().numpy()
            img2 = img2_batch[i].permute(1, 2, 0).cpu().numpy()
            label = int(labels[i].item())

            row_axes = axes[shown] if num_pairs > 1 else axes
            row_axes[0].imshow(img1)
            row_axes[0].axis("off")
            row_axes[0].set_title(f"Image 1 ({label})", fontsize=10)

            row_axes[1].imshow(img2)
            row_axes[1].axis("off")
            row_axes[1].set_title(f"Image 2 ({label})", fontsize=10)

            shown += 1
