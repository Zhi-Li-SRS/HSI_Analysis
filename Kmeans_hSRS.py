import argparse
import os
from pathlib import Path

import numpy as np
import tifffile
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


class hSRSAnalyzer:
    def __init__(
        self, base_folder, output_folder, n_clusters=8, wave_start=2700, wave_end=3150, wave_points=60
    ):
        self.base_folder = Path(base_folder)
        self.output_folder = Path(output_folder)
        self.n_clusters = n_clusters
        self.wavenumbers = np.linspace(wave_start, wave_end, wave_points)
        self.cmaps = np.array(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 1, 1]]
        )

    def load_and_preprocess(self, image_path):
        """Load and preprocess the image stack"""
        image = tifffile.imread(image_path)
        assert len(image.shape) == 3, "Image should be a hyperspectral image stack"
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return image

    def perform_kmeans(self, image):
        """Perform K-means clustering on the image"""
        spectra, height, width = image.shape
        reshaped_image = image.reshape(spectra, -1).T

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(reshaped_image)
        segmentation_map = labels.reshape(height, width)

        cluster_means = np.zeros((spectra, self.n_clusters))
        cluster_stds = np.zeros((spectra, self.n_clusters))

        for i in range(self.n_clusters):
            cluster_pixels = reshaped_image[labels == i]
            cluster_means[:, i] = np.mean(cluster_pixels, axis=0)
            cluster_stds[:, i] = np.std(cluster_pixels, axis=0)

        return segmentation_map, cluster_means, cluster_stds

    def save_segmentation_map(self, segmentation_map, output_path):
        """Save segmentation map directly as tiff file"""
        height, width = segmentation_map.shape
        rgb_image = np.zeros((height, width, 3))

        for i in range(self.n_clusters):
            mask = segmentation_map == i
            for c in range(3):
                rgb_image[mask, c] = self.cmaps[i, c]
        rgb_image = (rgb_image * 255).astype(np.uint8)
        tifffile.imwrite(output_path / "segmentation_map.tiff", rgb_image)

    def plot_results(self, folder_name, segmentation_map, cluster_means, cluster_stds):
        """Plot and save the segmentation map and spectral profiles"""
        output_path = self.output_folder / folder_name
        output_path.mkdir(parents=True, exist_ok=True)

        self.save_segmentation_map(segmentation_map, output_path)
        plt.figure(figsize=(10, 6))
        for i in range(self.n_clusters):
            color = self.cmaps[i]
            plt.plot(self.wavenumbers, cluster_means[:, i], color=color, label=f"Cluster {i+1}")
            plt.fill_between(
                self.wavenumbers,
                cluster_means[:, i] - cluster_stds[:, i],
                cluster_means[:, i] + cluster_stds[:, i],
                color=color,
                alpha=0.2,
            )

        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Normalized Intensity")
        plt.title("Cluster Spectral Profiles")
        plt.legend()
        plt.savefig(output_path / "spectral_profiles.tif", dpi=300, bbox_inches="tight")
        plt.close("all")

    def process_dataset(self):
        for folder in self.base_folder.iterdir():
            if folder.is_dir():
                print(f"Processing {folder.name}...")
                image = self.load_and_preprocess(folder)
                segmentation_map, cluster_means, cluster_stds = self.perform_kmeans(image)
                self.plot_results(folder.name, segmentation_map, cluster_means, cluster_stds)


def get_args():
    parser = argparse.ArgumentParser(description="Hyperspectral SRS Image Analysis")
    parser.add_argument(
        "--base_folder", type=str, required=True, help="Base folder containing the image data"
    )
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for results")
    parser.add_argument(
        "--n_clusters", type=int, default=6, help="Number of clusters for K-means (default: 8)"
    )
    parser.add_argument("--wave_start", type=float, default=2700, help="Starting wavenumber (default: 2700)")
    parser.add_argument("--wave_end", type=float, default=3150, help="Ending wavenumber (default: 3150)")
    parser.add_argument(
        "--wave_points", type=int, default=60, help="Number of wavenumber points (default: 60)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    analyzer = hSRSAnalyzer(
        base_folder=args.base_folder,
        output_folder=args.output_folder,
        n_clusters=args.n_clusters,
        wave_start=args.wave_start,
        wave_end=args.wave_end,
        wave_points=args.wave_points,
    )
    analyzer.process_dataset()
