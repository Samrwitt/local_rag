# Obstacles and options

Access is treated as allowed for the user who runs the app. Selected-file indexing is v1; folder auto-crawl is the scale-up path.

## Selected files → auto crawl

Indexing everything at once is slow (OCR on every photo). Starting with selected files keeps demos and reviews honest. Folder crawl is already implemented for later authorized auto-index.

## OCR is not raw storage

Cafe names and street lines must become city/country/language/roles. Underscores in names (`Italy_2019_rome.mp4`) do not create word boundaries, so matching normalizes `_` and `-` to spaces. City lists are finite; unknown towns need the list extended or a geocoder later.

## Video: metadata only

Frames and speech are out of scope. Container tags plus the file name carry date and place. A rename like `Italy_2019_rome_clip.mp4` is enough to join a trip query; a generic `clip.mp4` with no tags will not.

## Trip queries

“Italy 2019” should return receipt + hotel + memo + video, not three chunks of one note. Search keeps the best chunk per file first, then soft-boosts matching country/year/city. Document dates inside the text beat file modified time when present.

## Language cues

Shared words like “hotel” are weak. Language is guessed from stronger local cues, then from country when needed.

## Model answers

Small models can invent dates. Local preview always shows the passages. Prefer a model that stays on the sources when the model step is on.
