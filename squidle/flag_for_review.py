# This assumes you have installed sqapi in the environment that you're working in
# pip install sqapi
from sqapi.api import SQAPI
import json

# This object simplifies interactions and authentication with the API
api = SQAPI()

# list of annotation.ids to update
annotation_ids = []  # add in all annotation ids that you'd like to update
annotation_set_id = 0  # add the annotation_set id that you'd like to update

# Build up the qs params
q = dict(filters=[dict(name="annotation_set_id", op="eq", val=annotation_set_id),
                  dict(name="id", op="in", val=annotation_ids)])

# a good sanity check before running the patch is to do a get.
#r = api.get(f"/api/annotation", qsparams={"q": json.dumps(q)}).execute().json()
#print(r)

# To update the labels, uncomment one of the options below

# # OPTION1: Update all at once in batch
# r = api.patch(f"/api/annotation", qsparams={"q": q}, json_data=dict(needs_review=True, description="Flagged for review by Kelham's bot")).execute().json()

# # Option 2: Update one at a time, note this does not have an annotation_set.id filter. Only uses the annotation.ids
for annotation_id in annotation_ids:
     api.patch(f"/api/annotation/{annotation_id}", json_data=dict(needs_review=True, comment="Flagged for review by Urchin detector AI")).execute()

