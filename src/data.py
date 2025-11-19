import json
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from glob import glob

from utils.utils import video_abs_path

DEBUG = False

@dataclass
class ConversationSample:
    messages: List[Dict[str, Any]]


def format_mcq_question(question: str, choices: Optional[list]) -> str:
    """Combine question and choices into a single instruction-friendly string."""
    base_instr = (
        "You are an AI assistant specialized in Vietnam traffic senario understanding."
        "Answer the multiple choice question based on the video provided. "
        "Please select one of the provided choices and respond with only the "
        "choice letter in A, B, C, or D."
    )
    if choices:
        return f"{base_instr}\n\n{question} {' '.join(choices)}"
    return f"{base_instr}\n\n{question}"

def build_user_content(
    video_path: str, question: str, choices: Optional[list], use_unsloth: bool, signs: Optional[list] = None
) -> dict:
    """Construct user content for conversation sample based on format."""
    video_path = video_abs_path(video_path)
    question = format_mcq_question(question, choices)

    if use_unsloth:
        content = {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "resized_height": 1080, "resized_width": 1920},
                {"type": "text", "text": question},
            ],
        }
        if signs:
            for sign in signs:
                content["content"].append({"type": "image", "image": sign})
            content["content"][1]['text'] += "\n\nRefer to the provided traffic sign images when answering the question if relevant."
    else:
        # Generic processor format (with metadata structure) if not using Unsloth
        content = {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {"video_path": video_path, "fps": 1, "max_frames": 128},
                },
                {"type": "text", "text": question},
            ],
        }
    if DEBUG:
        print("Built user content:", content)
    return content

class RoadBuddyVideoDataset(Dataset):
    """Dataset converting train.json rows into Unsloth vision conversation format.

    Each item becomes a list of messages: user (video + text) then assistant (answer text).
    Unsloth expects: [{role: user, content: [...]}, {role: assistant, content: [...]}]
    Video is referenced by path; reading/decoding is handled later by vision_utils.
    """

    def __init__(self, data_path: str, root_dir: str = "data", use_unsloth: bool = True, signs_dir: Optional[str] = None):
        with open(data_path, "r") as f:
            raw = json.load(f)
        self.items = raw["data"]
        self.root_dir = root_dir
        self.use_unsloth = use_unsloth
        self.signs = glob(os.path.join(signs_dir, "*")) if signs_dir and os.path.exists(signs_dir) else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ConversationSample:
        item = self.items[idx]

        user_content = build_user_content(
            item["video_path"], 
            item["question"], 
            item['choices'], 
            self.use_unsloth, 
            signs=self.signs
        )

        raw_ans = item.get("answer", "")
        # Target should be just the choice letter (A/B/C/D)
        if isinstance(raw_ans, str) and "." in raw_ans:
            answer_text = raw_ans.split(".", 1)[0].strip()
        else:
            answer_text = (raw_ans or "A").strip()[:1]

        sample = [
            user_content,
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        ]
        return ConversationSample(messages=sample)