import unittest
from unittest.mock import patch

import app
from preprocessing import preprocess


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_stemming_and_stopword_order(self):
        self.assertEqual(
            preprocess(
                "PEMERINTAH mengumumkan pembangunan jalan baru! "
                "https://contoh.id/berita 123"
            ),
            "perintah bangun jalan",
        )
        self.assertEqual(
            preprocess("Para pemimpin menyerang kelompok tersebut."),
            "pimpin serang kelompok",
        )
        self.assertEqual(preprocess("dan yang di"), "")

    def test_endpoint_transforms_cleaned_text_and_matches_model(self):
        text = "Para pemimpin menyerang kelompok tersebut."
        vector = app.tfidf.transform(["pimpin serang kelompok"])
        expected = {
            "prediction": int(app.model_ros.predict(vector)[0]),
            "probability": float(app.model_ros.predict_proba(vector)[0][1]),
        }
        with patch.object(app.tfidf, "transform", wraps=app.tfidf.transform) as transform:
            response = app.app.test_client().post("/predict", json={"text": text})
        transform.assert_called_once_with(["pimpin serang kelompok"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), expected)

    def test_endpoint_uses_shared_preprocessor(self):
        self.assertIs(app.preprocess, preprocess)


if __name__ == "__main__":
    unittest.main()
