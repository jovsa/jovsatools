# jsmltools
> learning -> production


## Background
* SCQ
* [my physical notes on this](https://photos.app.goo.gl/yaggyEt4fdD52Xw6A)
* everthing will be WIP (by definition)

## Goals
* Compounding effects with learning/exeprimentation
    * (open ?): How to track? Is it worth tracking?
* Quick portability
    * for sandboxing
    * so this becomes the canonical repo for experimentation/learning
* Multi-domain compatibity
* Incrementability
    * Don't want to spend too much time on `setup` and `boilerplate`
    * Allow for quick ideas/initial exploration to stable APIs to co-exist

## Design Choices
* Using [nbdev](https://github.com/fastai/nbdev) as a base structure
    * (+) Will use this to get something off the ground. However, will not tie too deeply into this strcuture (if I need to swap this out).
    * (+) Nice literate env
    * (+) Get PyPI functionality out of the box
    * (+) Get docs functionality out of the box
    * (+) Get unit test functionality out of the box
    * (-) Might need to contribute to nbdev to add functionality
        * (mitigation) clone an internal verison of nbdev
        * (mitigation) write all non-supported use case directly in the `lib` folder
* Monolithical git repo
    * (+) Will allow for incrementability
    * (+) Will become canonical place with experimentation/learning will happen. This will help with compounding learning/experimentation speed.
    * (-) Will enentually get hard to maintain?
        * (mitigation) things don't have a lot of deps will eventually fall out/depricate
    * (-) >0 startup time
* Lack of structure
    * (+) [Hyrum's Law](https://www.hyrumslaw.com/)
    * (-) hard for others to understand/follow
        * (migtigation) build tools to show dependency
    * (+) allow for 
* [Progressive disclosure of complexity](https://twitter.com/fchollet/status/1231285340335267840)
    * (+) easy to start
    * (+) allow for rough initial ideas to stable APIs to co-exist
    * (-) A lot of extra work needed to trace dependencies
        * (mitigation) build better tools to view dependencies
* Multi-language/multi-framework/multi-domain support.
    * (+) This becomes the canonical repo for learning/experimentation.
    * (-) Only python will be used in notebooks (for now and because of nbdev limitation).
        * (mitigation): write all non-supported use case directly in the `lib` folder.

## Refrences

* https://notes.andymatuschak.org/
    * Metacognitive supports as cognitive scaffolding
        * by contraint and blog by narritive
    * [Incremental Writing](https://supermemo.guru/wiki/Incremental_writing)
    * Notes to APIs
    * Exponential growth to learn
    * Cross-decipline encouragement

