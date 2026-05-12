# Ideas for automated research pipeline


## Current working projects

### Same pipeline, new model

**Deliverable**: A skill (?) or MCP that reliably has Claude replicate a tentative result across different settings

Paradigmatic Use case: We found that the FRA gets a very big effect in Qwen-14B. It sure would be nice to know if it replicates across all other 

Paradigmatic counter-example: I want to know if this effect generalizes across all layers. Can we make that measurement? 
Paradigmatic counter-example: Spinning up an agent to launch a command on the command line.

Does this need to exist?
Similar to many other automating proposals, it's worth asking whether a tool _needs_ to exist for this, or whether this is a symptom of an underlying process problem.




### Replication

**Deliverable** A skill (?) or MCP that reliably has Claude replicate an existing result

Use case: "I want to see if our techniques works on EM. Can you first replicate their finding, and then launch an agent to do X?"


### Night run/Take-over

Paradigmatic use case: Claude can you keep running this experiment overnight, fix any bugs and let me know how it goes?
Paradigmatic use case: I'm hopping out for lunch, can you takeover and keep going on your own?


Notes:
Lots of important things here: 
1. Hardware:
    - Can the Claude's decide how much to accelerate? I think yes, but you should set a default spend limit, and have an override that lets you increase it. Probably good to have a hard limit you can't exceed. This would be nice because then the Claude's could spin up, say 20x A40's and 
    - Storage is important. I think it would be good to have on persistent bucket where you can chuck everything into. Not sure if AWS or HF is the solution.
    - 
### Multi-Model Review

Not sure what the best way to do this, but it would be awesome if you had a skill that let's Claude call other models.

Paradigmatic use case: Here is my abstract. Can you spin up a debate protocol between three different models to refine and improve it?


### Dashboard (special case)

In the W2S automated researcher, the human-operator surface is a web dashboard (Flask server at `localhost:8000`): queue runs, watch a leaderboard, read a shared "findings forum" where parallel agents post intermediate results. We don't want this as the *default* interface — local Claude Code + MCP is — but we do want it as a **special-case opt-in**: a single `/dashboard` skill/module that stands up the analogue (web UI over the same controller state) for projects where a glanceable dashboard beats prompting. Treat it as one consumer of the controller's API, not the primary one.





