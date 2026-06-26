import re
import os
import csv
import ast
import html
import unicodedata
class input_preprocessor_1:
    def rationalize(self, inp, nam1, outp, nam2):
        try:
            pt1 = os.path.join(inp, nam1)
            pt2 = os.path.join(outp, nam2)

            out_dir = os.path.dirname(pt2)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            def fix_text_artifacts(text):
                if text is None:
                    return ""

                text = unicodedata.normalize("NFKC", str(text))
                text = text.replace("\x00", "")
                text = text.replace("<unk>", "")
                text = text.replace("â€™", "'")
                text = text.replace("â€œ", '"')
                text = text.replace("â€", '"')
                text = text.replace("â€”", "—")
                text = text.replace("â€“", "–")
                text = text.replace("â€", "")
                text = text.replace("�", " ")

                if "Ã" in text or "â" in text:
                    try:
                        repaired = text.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
                        if repaired.count("Ã") < text.count("Ã") or repaired.count("â") < text.count("â"):
                            text = repaired
                    except Exception:
                        pass

                previous = None
                while text != previous:
                    previous = text
                    text = html.unescape(text)

                return text

            def normalize_text(text):
                if text is None:
                    return ""

                text = fix_text_artifacts(text)

                def normalize_segment(segment, preserve_code=False):
                    if preserve_code:
                        segment = segment.replace("\r\n", "\n").replace("\r", "\n")
                        return segment.strip()
                    segment = segment.replace("\\r", " ")
                    segment = segment.replace("\\n", " ")
                    segment = segment.replace("\\t", " ")
                    segment = re.sub(r"\s+", " ", segment)
                    return segment.strip()

                if "```" in text:
                    pieces = re.split(r"(```.*?```)" , text, flags=re.DOTALL)
                    normalized = []
                    for piece in pieces:
                        if piece.startswith("```") and piece.endswith("```"):
                            normalized.append(normalize_segment(piece, preserve_code=True))
                        else:
                            normalized.append(normalize_segment(piece))
                    return " ".join(p for p in normalized if p)

                return normalize_segment(text)

            def parse_messages(raw_messages):
                if raw_messages is None:
                    return []

                s = fix_text_artifacts(raw_messages)
                s = s.replace("\r", " ").replace("\n", " ")
                s = re.sub(r"}\s*{", "}, {", s)

                if not s.startswith("["):
                    s = "[" + s
                if not s.endswith("]"):
                    s = s + "]"

                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    return []

                if not isinstance(parsed, list):
                    return []

                cleaned = []
                for item in parsed:
                    if not isinstance(item, dict):
                        return []
                    role = normalize_text(item.get("role", "")).lower()
                    content = normalize_text(item.get("content", ""))

                    if role not in {"user", "assistant"}:
                        return []
                    if not content:
                        continue

                    cleaned.append({"role": role, "content": content})

                return cleaned

            def is_valid_conversation(conv):
                if len(conv) < 2:
                    return False

                if conv[0]["role"] != "user":
                    return False

                for i in range(1, len(conv)):
                    if conv[i]["role"] == conv[i - 1]["role"]:
                        return False

                # Need at least one assistant reply
                if not any(x["role"] == "assistant" for x in conv):
                    return False

                return True

            def build_training_text(prompt, conv):
                prompt = normalize_text(prompt)

                # The prompt is duplicated as the first user message in this dataset.
                # Remove that duplicate so the model does not see the same text twice.
                if conv and conv[0]["role"] == "user" and normalize_text(conv[0]["content"]) == prompt:
                    conv = conv[1:]

                if not conv:
                    return ""

                parts = ["<bos>"]

                if prompt:
                    parts.append(f"<user> {prompt}")
                else:
                    # Fallback: use the first turn if prompt is missing
                    first = conv[0]
                    parts.append(f"<{first['role']}> {first['content']}")
                    conv = conv[1:]

                for item in conv:
                    parts.append(f"<{item['role']}> {item['content']}")

                parts.append("<eos>")
                return "\n".join(parts)

            cleaned_rows = 0
            skipped_rows = 0

            with open(pt1, "r", encoding="utf-8-sig", newline="") as inpf, \
                 open(pt2, "w", encoding="utf-8", newline="") as outf:

                reader = csv.DictReader(inpf)
                writer = csv.DictWriter(
                    outf,
                    fieldnames=["prompt_id", "prompt", "training_text"],
                    quoting=csv.QUOTE_MINIMAL
                )
                writer.writeheader()

                for row in reader:
                    try:
                        prompt_id = normalize_text(row.get("prompt_id", ""))
                        prompt = normalize_text(row.get("prompt", ""))
                        conv = parse_messages(row.get("messages", ""))

                        if not prompt or not conv:
                            skipped_rows += 1
                            continue

                        if not is_valid_conversation(conv):
                            skipped_rows += 1
                            continue

                        training_text = build_training_text(prompt, conv)

                        if len(training_text) < 20:
                            skipped_rows += 1
                            continue

                        writer.writerow({
                            "prompt_id": prompt_id,
                            "prompt": prompt,
                            "training_text": training_text
                        })
                        cleaned_rows += 1

                    except Exception:
                        skipped_rows += 1
                        continue

            print(f"Mission successful. Cleaned rows: {cleaned_rows}. Skipped rows: {skipped_rows}.")

        except Exception as e:
            print(f"Mission failed... Error: {e}")
    def clipper(self,inp,nam1,outp,nam2):
         try:
            pt1=os.path.join(inp,nam1)
            pt2=os.path.join(outp,nam2)
            with open(pt1,"r",encoding="utf-8",newline="") as inpf, \
                 open(pt2,"w",encoding="utf-8",newline="") as outf:
                 reader=csv.reader(inpf)
                 writer=csv.writer(outf)
                 header=next(reader)
                 writer.writerow(header)
                 for i, row in enumerate(reader):
                      if i>=10000:
                           break
                      writer.writerow(row)
            print("Mission successful!")
         except Exception as e:
            print(f"Mission failed... Error: {e}")

if __name__ == "__main__":
    import sys
    obj = input_preprocessor_1()
    if len(sys.argv) == 1:
        obj.rationalize(r"C:\Users\adith\OneDrive\Desktop\Transformer_Project\data",
                         "test_gen_phase1.csv",
                         r"C:\Users\adith\OneDrive\Desktop\Transformer_Project\data",
                         "test_gen_phase1_clean.csv")
    elif sys.argv[1].lower() in {"clip", "clipper"}:
        obj.clipper(r"C:\Users\adith\OneDrive\Desktop\Transformer_Project\data",
                    "test_gen.csv",
                    r"C:\Users\adith\OneDrive\Desktop\Transformer_Project\data",
                    "test_gen_phase1.csv")
    elif sys.argv[1].lower() in {"clean", "rationalize"}:
        obj.rationalize(r"C:\Users\adith\OneDrive\Desktop\Transformer_Project\data",
                         "test_gen_phase1.csv",
                         r"C:\Users\adith\OneDrive\Desktop\Transformer_Project\data",
                         "test_gen_phase1_clean.csv")
    else:
        print("Usage: python Input_preprocessor_for_wiki_dataset.py [clean|clip]")

        