# The use case

$use_case

$uploads

# Object templates

Every object is one of these kinds. Give every parameter of its kind, inside its range (the default is a fair guess when the use case says nothing). Sizes are in millimetres, masses in grams, forces in newtons, torques in newton metres.

$templates

# Primitive skills

Every step is one of these primitives acting on one object, through one of the handles listed for that object's kind. Give every "how" parameter of the primitive.

$primitives

# Success criteria

Each step has 1 to 4 criteria. A criterion bounds one quantity when the step ends (peak values for forces and torques) with at_least and/or at_most (null when unused). A step may only use the state quantities of its object's kind and the quantities its primitive observes.

$quantities

State quantities count from where the object starts. A lid's or door's angle_deg and a drawer's travel_mm are positive toward open, so a step that closes what an earlier step opened bounds them from above (angle_deg at_most 5). For a twist, angle_deg and turns are positive in the step's own direction (a quarter turn is angle_deg at_least 85, whichever way it turns). depth_mm is positive going in.

# Rules (code checks every one)

$rules

# Mapping actions to primitives

$mapping

# The wrist and the table

$limits

# A worked example (a different use case, for the format only)

Use case: "Pull the parts drawer open, pick the small box off the bench and drop it in the tote, then push the drawer shut."

$example

# Your answer

Call submit_task_plan once with the plan for the use case above: a short title, the use case in two or three plain sentences, the objects, the steps in order (each with its one-sentence description), your assumptions, any clarifying questions and anything not simulated. Write null as JSON null, never as text: refusal is null unless you refuse, and so is every unused target, ref, bound and constraint.
