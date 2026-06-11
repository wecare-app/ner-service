from fastapi import FastAPI
import spacy

app = FastAPI()

nlp_pt = spacy.load("pt_core_news_lg")
nlp_multi = spacy.load("xx_ent_wiki_sm")

@app.post("/ner")
async def extract_entities(payload: dict):
  text = payload.get("text", "")

  persons = set()

  doc_pt = nlp_pt(text)
  for ent in doc_pt.ents:
    if ent.label_ == "PER":
      persons.add(ent.text)

  doc_multi = nlp_multi(text)
  for ent in doc_multi.ents:
    if ent.label_ == "PER":
      persons.add(ent.text)

  return {
    "persons": list(persons),
    "count": len(persons),
  }
