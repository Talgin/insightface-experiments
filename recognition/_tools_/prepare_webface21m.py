import sys
import cv2
import argparse
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, "/insightface/python-package")

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


class WebFace21MProcessor:
    def __init__(self, input_dir, output_dir, gpu_id=0, align_size=112, det_size=(1024, 1024), det_thresh=0.2):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.gpu_id = gpu_id
        self.align_size = align_size
        self.det_thresh = det_thresh

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu_id >= 0 else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name="buffalo_l",
            root="/models",
            allowed_modules=["detection"],
            providers=providers,
        )
        self.app.prepare(ctx_id=gpu_id, det_size=det_size, det_thresh=det_thresh)

        self.stats = {
            "processed": 0,
            "aligned": 0,
            "failed": 0,
        }

    def _get_largest_face(self, faces):
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    def _detect_with_retry(self, img):
        faces = self.app.get(img)
        if faces:
            return faces

        h, w = img.shape[:2]
        retry_scales = [1.5, 2.0]
        for scale in retry_scales:
            new_w = int(w * scale)
            new_h = int(h * scale)
            up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            faces = self.app.get(up)
            if faces:
                for face in faces:
                    face.bbox /= scale
                    if hasattr(face, "kps") and face.kps is not None:
                        face.kps /= scale
                return faces

        return []

    def detect_and_align(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        faces = self._detect_with_retry(img)
        if not faces:
            return None

        face = self._get_largest_face(faces)
        if getattr(face, "kps", None) is None:
            return None

        return norm_crop(img, face.kps, image_size=self.align_size)

    def process_dataset(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        identity_folders = sorted([p for p in self.input_dir.iterdir() if p.is_dir()])
        pbar = tqdm(identity_folders, desc="Processing identities", unit="id")

        for identity_folder in pbar:
            out_dir = self.output_dir / identity_folder.name
            out_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(
                list(identity_folder.glob("*.jpg")) +
                list(identity_folder.glob("*.jpeg")) +
                list(identity_folder.glob("*.png")) +
                list(identity_folder.glob("*.JPG")) +
                list(identity_folder.glob("*.JPEG")) +
                list(identity_folder.glob("*.PNG"))
            )

            for image_path in image_files:
                self.stats["processed"] += 1

                aligned = self.detect_and_align(image_path)
                if aligned is None:
                    self.stats["failed"] += 1
                    continue

                out_path = out_dir / image_path.name
                cv2.imwrite(str(out_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.stats["aligned"] += 1

                if self.stats["processed"] % 500 == 0:
                    rate = 100.0 * self.stats["aligned"] / max(1, self.stats["processed"])
                    pbar.set_postfix(
                        processed=self.stats["processed"],
                        aligned=self.stats["aligned"],
                        failed=self.stats["failed"],
                        success=f"{rate:.1f}%",
                    )

        rate = 100.0 * self.stats["aligned"] / max(1, self.stats["processed"])
        print(f"processed={self.stats['processed']}, aligned={self.stats['aligned']}, failed={self.stats['failed']}, success={rate:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--align_size", type=int, default=112)
    args = parser.parse_args()

    WebFace21MProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        align_size=args.align_size,
        det_size=(1024, 1024),
        det_thresh=0.2,
    ).process_dataset()


if __name__ == "__main__":
    main()