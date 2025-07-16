Rendering validation comes last, since it's the most expensive

We don't even need to render the videos directly, since this would take up quite a bit of space. It's cool to think about the code being a latent, compressed version of the video, and the type of intelligent system it takes to uncompress this into it's most elegant form. 


How should we handle custom assets? Should we just replace them with boxes? should we try to give the model access to a fixed database of assets and teach it how to use them like any other tool?
One idea:
Context-aware replacement based on asset type:
- Characters → Stick figures or simple geometric representations
- Icons → Relevant Manim shapes (factory → Rectangle with smaller rectangles on top)
- Mathematical symbols → Use Manim's MathTex
- Backgrounds → Solid colors or gradients
- Photos → Rectangle with text label

The philosophy is "fix don't reject" since some issues can be fixed programatically. This keeps more data while ensuring it's at least syntactically valid and follows basic Manim
  structure.

# Code Cleaning and Formatting
There is basically 2 types of datasets here: existing databases of supposedly precleaned manim fine tuning data, and datasets that we are creating ourselves by web scraping and downloading github repos.

For the first type, we just need to make sure that the other data sources don't have any of the exact same code

# TODO writeup your analsis of the different sources and their overlap, as well as your new sources

need to ensure that the code is actually valid python,

Inlining needed on al

# Why this is hard
Combining the different existing datasets was actually just a matter of getting the deduplication and formatting right, but getting data from the untapped sources is a different beast. 

- Assets: vaguely stuck between a rock and a hard place here. The best solution would probably be to chop the whole video into snippets, but this proves tricky in its own way.
- Inlining and code scattered in the repo
- Utility functions and imports
- Special packages like opencv (we just installed these)





# Distribution
Do we properly cover the distribution of lengths?
Are we able to do any curriculum learning use the length as a super cheap-to-compute proxy for difficulty?

When we get long form videos, are we trying to split them into self contained scenes? If so, there is the issue of the scenes being interdependent, which is actually not good for the dataset, since we are teaching the model to write code that references variables that are not there. 

# Gotchas
- inlining large files as string literals might change how Manim's `Write()` animation behaves, even with the same `run_time` parameter