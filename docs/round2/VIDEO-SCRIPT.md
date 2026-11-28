# Round 2 demo video: shot list and words

Target 3:45, hard limits 3:00 to 5:00. 1920 x 1080, 30 fps, one browser window, cursor visible, notifications off.
Spoken by the owner in his own voice: record it yourself, in one or two takes per line, and do not use a synthetic
voice. The organisers asked for this explicitly, and a synthetic read is the first thing a judge notices.

The order below is the order the brief asks for: the space being captured on a phone, the twin in the browser, places
tagged live, a search that finds a place, a route drawn on the plan, and the camera walking that route. Narration
covers the technical approach in shots 3 and 4 and the walkthrough in shots 5 to 7, which is the other thing the brief
asks for.

## Before you record

| # | What |
|---|---|
| 1 | **Capture footage, 10 s.** A 10 second cut from the real chapel clip (`DJI_20260912135353_0051_D.MP4`), or someone filming you holding the phone and walking. It must be the actual capture, not a re-enactment of one. |
| 2 | **Twin preloaded.** `https://ricco.codes/ply/walk.html?scene=https%3A%2F%2Fply-api.ricco-ply.workers.dev%2Fv1%2Fscenes%2Fchapel-q1b-1%2Fscene.json`. Load it once, wait for the splat to settle, then reload the page before the take so the first frame is not a blur. |
| 3 | **A second tab with the same twin** for the route shot, already walked once so the router has run. |
| 4 | **Tabs in this order:** twin, the twin again with the map opened large, the capture guide at `/ply/guide.html`, `ab.html` (the two twins side by side) in case a comparison is wanted. |
| 5 | **Search term.** `altar` is a place in this twin and it sits at the far end, so the route is worth watching. Rehearse it once; do not type it live for the first time. |
| 6 | **Camera and audio.** Phone on a tripod or a laptop mic close to you, quiet room, no fan. Record picture and voice separately if you can, so a cough costs one line and not the take. |
| 7 | **Screen.** 100% browser zoom, bookmarks bar hidden, do not disturb on, cursor still except when clicking. |
| 8 | **A copy of the numbers** in front of you: 1,223 poses, 0.58 px, 15,000 steps, 281 places, 38.6 MB twin, 20.4 MB phone file. Read them, do not recall them. |

## Shots

### 1. 0:00 to 0:12. The space, being captured on a phone

Picture: the 10 second capture cut, handheld, the chapel aisle and pews going past. Then a 2 second freeze on one frame
with the A4 marker visible.

> "This is a chapel in Bombay, filmed on a phone, one walk, eight clips. No scanner, no floor plan, no laser. That is
> the whole input. Everything you are about to see comes out of this footage."

Say it flat and slow. No music under the first shot, or a quiet room tone only.

### 2. 0:12 to 0:30. What Ply is

Picture: cut to the twin in the browser, standing in the aisle, and hold still.

> "Ply turns that walk into a twin you can walk around in a browser, with the things in it labelled, a search over
> those labels, and a route to any of them that goes around the furniture. This round is about those three: places,
> search, routing."

### 3. 0:30 to 1:05. The pipeline, in numbers

Picture: screen capture of `report.json` scrolling, or a simple slide of the stage list, then back to the twin.

> "Here is what happens to the video. It keeps the sharpest frame every four tenths of a second: twelve hundred and
> thirty six frames from eight clips. Structure from motion solves the camera for twelve twenty three of them, at about
> half a pixel of reprojection error. It takes gravity from the phone, fits the floor, and scales the room from the
> height I held the phone at, one point four five metres. Then it builds a walking grid at ten centimetres, labels the
> objects with SAM 3, and trains the splat for fifteen thousand steps on one GPU, which billed ninety two cents."

Point at the numbers on screen as you say them. If the on-screen text does not match a number, fix the screen, not the
sentence.

### 4. 1:05 to 1:30. The twin, in the browser

Picture: walk forward down the aisle with W, turn with the mouse, then click the minimap so it opens large.

> "This is the twin. Thirty eight megabytes, twenty on a phone, and it is streaming, so it opens on a laptop without a
> plugin or an install. I walk it like a game. That map in the corner is not a drawing someone made: it is a picture of
> the twin from above, cut at one point eight metres, and the grid underneath it is the one the router plans on."

### 5. 1:30 to 2:05. Places, tagged

Picture: press `Ctrl`+`K` to open the palette, close it, then let the pins show. Hover two: a door and a pew. Open the "Found
by AI" group so the class and the percentage are readable.

> "Two hundred and eighty one places in this room, found automatically. Each one has a class, a position and a
> confidence, and the confidence is on the row as I hover it, so you can see what the detector was sure about and what
> it was not. The ones worth pinning, doors, stairs, the altar, stained glass, show as pins as soon as the twin loads.
> The rest are searchable but stay out of the way, because two hundred and eighty one floating labels is a mess."

Then, if the design lab is being shown:

> "And a person can overrule it. This is the review list: tick the ones that are right, dismiss the ones that are not,
> and the twin keeps the difference."

Say the last line only while the review list is on screen. It runs on the demo capture, not on the chapel, so do not
imply the chapel's places were edited by hand in this take.

### 6. 2:05 to 2:35. Search finds a place

Picture: `Ctrl`+`K`, type `alt` slowly, arrow down, Enter. The route card appears.

> "I want the altar. I type three letters, it is the first hit, and the card gives me the distance and the turns. There
> is no cloud call here: the place list ships with the twin, about seventy kilobytes, and the search is a filter over
> it, so it works on a bad connection and it answers instantly."

Read the distance off the card as it appears. Do not guess it.

### 7. 2:35 to 3:20. The route on the map, then walking it

Picture: the enlarged map with the route drawn on it, hold 4 seconds so the line around the pews is visible. Back to the
walkthrough, press "Take me there", and let the camera walk 8 to 10 seconds.

> "Here is the route on the plan. It is A star on that ten centimetre grid, and the walker is a circle twenty two
> centimetres across, so the route keeps that much clear of every wall and every pew. Watch the line: it does not cut
> across the pews, it takes the aisle and bends where the aisle bends. Unknown cells count as blocked, so it will not
> send me through a wall it never saw, even if that means a longer way round."

Then, while the camera walks:

> "Take me there, and it walks the route itself. I can take over with W A S D at any point, and if I wander off it
> replans from where I am."

Say "it does not cut across the pews" only if the drawn line visibly avoids them in the take. If it clips a corner,
that is a bug worth showing and worth fixing before the video, not worth narrating around.

### 8. 3:20 to 3:45. What is not finished, and close

Picture: back to the wide aisle view, then the end card with the links.

> "What is not finished: you cannot add your own place to a twin yet, by clicking and naming it. That is next. The mesh
> is not good enough to ship, and one twin is one space, so a building of rooms and floors is still separate twins. The
> capture is a walk of a few minutes, the twin takes about ninety minutes of machine time, and a run bills under a
> dollar. Links are in the master document."

End card, 4 seconds, no voice: `ricco.codes/ply`, the repo, and the master links PDF.

## Timing

| Shot | Length | Words | What it has to show |
|---|---:|---:|---|
| 1 Capture | 12 s | 32 | real footage, phone in hand |
| 2 What Ply is | 18 s | 45 | the twin, still |
| 3 Pipeline | 35 s | 96 | numbers on screen |
| 4 Twin | 25 s | 63 | walking, and the plan |
| 5 Places | 35 s | 92 | pins, classes, confidences |
| 6 Search | 30 s | 76 | typing, first hit, route card |
| 7 Route | 45 s | 108 | route on the map, then the walk |
| 8 Close | 25 s | 65 | limits, then the end card |
| **Total** | **3:45** | **577** | |

577 words in 3 minutes 41 seconds of speech is about 156 words a minute, which is a normal talking pace. If the take
runs long, cut shot 3 by ten seconds before touching shot 7: the route is the thing this round is judged on.

## If free-form tagging lands before 30 November

The version above shows what exists today: labels from the labelling stage, pins, and a person accepting or dismissing
a detection. If clicking the twin and naming a place is in the build by the deadline, replace shots 5 and 6 with it and
change two lines:

- Shot 5 becomes: "I click the bench, type 'choir bench', and it is a place now. Two taps and a name. It stays with the
  twin, so the next person who opens it can search for it."
- Shot 6 keeps its words but drops "about seventy kilobytes" if the place list is now written back to the scene.

Record the new shot first, in the same room and the same light, so the two halves match.
