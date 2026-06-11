from fastapi import FastAPI
import spacy

app = FastAPI()

EXCLUDE = ["tagger", "parser", "lemmatizer", "morphologizer", "attribute_ruler"]
nlp = spacy.load("pt_core_news_sm", exclude=EXCLUDE)

@app.post("/ner")
async def extract_entities(payload: dict):
  text = payload.get("text", "")
  persons = {ent.text for ent in nlp(text).ents if ent.label_ == "PER"}
  return {
    "persons": list(persons),
    "count": len(persons),
  }
