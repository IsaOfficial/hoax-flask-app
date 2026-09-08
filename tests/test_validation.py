import json
import unittest
from unittest.mock import patch

import app


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def assert_invalid(self, response):
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.is_json)
        data = response.get_json()
        self.assertIsInstance(data["error"], str)
        self.assertTrue(data["error"])
        self.assertNotIn("prediction", data)
        self.assertNotIn("probability", data)

    def test_invalid_payloads_do_not_reach_model(self):
        payloads = [
            None, [], "berita", 123, True, {},
            {"text": None}, {"text": 123}, {"text": True},
            {"text": []}, {"text": {}}, {"text": ""},
            {"text": " \t\n "}, {"text": "123 !!!"},
            {"text": "https://contoh.id/berita"}, {"text": "dan yang di"},
        ]
        with patch.object(app.model_ros, "predict") as predict, \
                patch.object(app.model_ros, "predict_proba") as probability:
            for payload in payloads:
                with self.subTest(payload=payload):
                    self.assert_invalid(self.client.post(
                        "/predict", data=json.dumps(payload),
                        content_type="application/json",
                    ))
            predict.assert_not_called()
            probability.assert_not_called()

    def test_malformed_json_and_wrong_content_type(self):
        for body, content_type in [
            ('{"text":', "application/json"),
            ("", "application/json"),
            ('{"text":"berita"}', "text/plain"),
            ("text=berita", "application/x-www-form-urlencoded"),
        ]:
            with self.subTest(body=body, content_type=content_type):
                self.assert_invalid(self.client.post(
                    "/predict", data=body, content_type=content_type,
                ))

    def test_valid_text_with_surrounding_whitespace(self):
        text = "Para pemimpin menyerang kelompok tersebut."
        normal = self.client.post("/predict", json={"text": text})
        padded = self.client.post("/predict", json={"text": " \n" + text + "\t "})
        self.assertEqual(normal.status_code, 200)
        self.assertEqual(padded.status_code, 200)
        self.assertEqual(normal.get_json(), padded.get_json())


if __name__ == "__main__":
    unittest.main()
