# Studio templates and primitives in physics (track S2)

**Product mode. Not part of the pre-registered experiment.** This file documents track S2 of the
Effector Lab Studio ([STUDIO.md](STUDIO.md), contract `rtt-effectors/2`): the six articulated
object templates and the real controllers and success predicates of the seven primitives. The
site never renders it.

Code: `src/rtt/studio/templates/{hinged_lid,drawer,quarter_turn_latch,threaded_joint,handwheel_valve,port_plug}.py`
and `src/rtt/studio/primitives/`. Tests: `tests/test_studio_templates.py`,
`tests/test_studio_primitives.py`.

Numbers marked [assumed] are modelling choices with no source; they are the same for every
candidate effector, and each one is a named constant in its module.

## Contents

1. [Rules every controller follows](#1-rules-every-controller-follows)
2. [Templates](#2-templates)
3. [Primitives](#3-primitives)
4. [Equal budgets](#4-equal-budgets)
5. [Fidelity and caveats](#5-fidelity-and-caveats)
6. [Gate results](#6-gate-results)
7. [Known limits](#7-known-limits)
8. [S2b: the hero-task limits](#8-s2b-the-hero-task-limits)
9. [S2c: tool stacks and the friction torques](#9-s2c-tool-stacks-and-the-friction-torques)
10. [S2d: every goal of a step](#10-s2d-every-goal-of-a-step)

## 1. Rules every controller follows

* **Perception.** A controller locates objects only through `StepContext.observe_site` (the
  true site plus the object's hidden aim offset). It never reads the true object state: the
  template quantities it needs (an angle, a travel, a depth) come from observed sites
  (`_common.observed_state`, `panel_angle`), so the placement error stays unknown to it, as in
  v1. A test (`test_controllers_never_read_the_true_object_state`) scans the package. The
  predicates read the true state.
* **Proprioception and force sensing.** A controller may read its own wrist joints and the
  contact forces between the tool and the objects (a wrist force sensor). Force feedback
  within a step acts through the wrist targets and the effector controls only, with the equal
  per-step budgets of section 4 (orchestrator ruling on STUDIO.md open question 1).
* **The tool's own mechanism.** Every effector control comes from the head's own v1 primitive,
  `command(t, phase, closure, model, data, V1SceneView)`, driven by the controller's closure
  profile (0 open, 1 closed). Fingers close on the synergy ramp, a cup switches its vacuum on only
  while its seal test holds, a soft arm pulls its cable: exactly as in the v1 layer. A primitive
  never adds grip of its own.
* **Tool geometry** comes from a calibration run (`_common.footprint`): the active head alone,
  0.75 m above the table, its controls held at a closure for 0.6 s, its collision geoms read in
  the wrist frame. It gives the tip, the aperture between fingers along the closing axis (for
  pre-shaping to a feature with 6 mm clearance each side), where a pair of pads mounted off the
  TCP meets a part (`grip_centre`), the tool's sideways hook (`hook`: the geoms near the tip that
  reach furthest beyond the rest, a soft arm's horizontal arm) and the sides a push can use. It
  never touches the step's `MjData` and is cached per compiled scene.
* **Wrist.** The palm yaw and the continuous roll turn about the same vertical axis. Controllers
  set headings with the yaw and turn objects with the roll; the roll target never moves faster
  than 98 % of the 180 deg/s cap [assumed, STUDIO.md 4.1]. The roll torque limit is 15 N m
  [assumed]. Free-space moves are min-jerk at 0.4 m/s peak [assumed]; transits rise to a height
  where the tool tip (and anything carried) clears every object by 30 mm.
* **Contacts counted as the tool's.** Force, penetration and guarded moves use every tool-side
  collision geom (all heads, the bracket or changer, and the wrist flange): a wrist that presses
  on an object is the tool pressing on it.
* **Guarded moves.** A final approach onto a feature stops advancing while the tool presses on
  any object harder than 5 N (S2b: not only the step's object, so a hook moving in under a lid
  stops at the bin's wall).
* **Every goal of a step** (S2d). Each at_least criterion is a goal, and a controller pursues
  them all: it ends when every one is met, or when a limit ends it (an at_most bound, the step's
  `max_force_n` or `max_torque_nm`, slip, a stall, the budget), never at the first goal reached.
  Several criteria on one quantity all hold: the largest at_least and the smallest at_most
  (`_common.criterion_bounds`). A force or torque goal is met once the measured force or torque
  has been 3 % over it for 0.03 s (three check ticks) [assumed], so the predicate's samples see
  it; a `force_n` at_most limits a push or a pull like `max_force_n` (80 % of it).

## 2. Templates

Every template is built in its instance frame (the pose on the table, turned by `yaw_deg`),
names every element `obj_<instance>_`, gives every geom `priority = 1` (the object's friction,
jittered by the seed, governs every tool contact), and keeps passive behaviour in its own joints
or actuators (`TemplateBuilder.passive`). Dry frictions on template joints use a near-hard
friction constraint (`solimp 0.999 0.9999 0.0001`) so a load below the friction does not creep.
Auxiliary sites beyond the handles exist only for controllers to observe geometry.

### 2.1 `hinged_lid` (rigid)

| Item | Model |
|---|---|
| Lid | horizontal panel `width x depth x thickness`, hinge along its back edge at mid-thickness, closed underside **120 mm** above the table (the default bin height, so a default lid at a default bin's pose covers it) [assumed: no hinge-height param], a static back wall carries it; gravity acts on it (it stays open only past vertical) |
| Door | vertical panel `depth` wide, `width` tall, bottom 10 mm above the table [assumed], hinge on its left edge, opens toward its front (local -y); a static jamb behind the hinge line |
| Joint | hinge `[0, max_angle_deg]`, damping `damping_nms`, armature 0.0005 |
| `open_torque_nm` | a catch near closed: the template's motor actuator applies that torque toward closed up to 6 deg, fading to 0 at 12 deg [assumed shape] |
| `feature = lip` | `handle` site at the middle of the free edge, mid-thickness (a door's: 40 mm below its top); a door's stop post behind the panel sits `feature_mm` in from the free edge (60 % of the door height) |
| `feature = handle` | a 10 mm bar [assumed] on two posts, `feature_mm` clear of the outer face: a lid's along x, 20 mm in from the front edge; a door's vertical, 25 mm in from the free edge, on its upper part |
| Sites | `handle`, `lid` (outer-face centre), `hinge` (aux, static) |
| State | `angle_deg` (0 closed, positive open), read within `[0, max_angle_deg]`: the solver's soft limit lets a panel resting on its stop lean a few thousandths of a degree past it |
| Density | 700 kg/m^3 [assumed] |

### 2.2 `drawer` (rigid)

| Item | Model |
|---|---|
| Carcass | static box `width x depth x height`, 10 mm walls [assumed], open at the front |
| Drawer | an overlay front `width x height` (12 mm) just in front of the carcass and a box in it with 2 mm clearance [assumed]; one slide along local -y, range `[0, travel_mm]`, friction loss `pull_force_n`, damping 2 N s/m [assumed] |
| Handle | centre `min(height/2, 30 mm)` below the top edge: `bar` 10 mm x 120 mm (or 60 % of the width) on posts, `handle_mm` off the face; `knob` 12 mm neck `handle_mm` long with a 26 mm head; `lip` a 4 mm plate standing `handle_mm` up from the front's top edge [assumed geometry] |
| Sites | `drawer` (front centre), `handle`, `front` (aux, carcass front) |
| State | `travel_mm`, read within `[0, travel_mm]` |
| Note | the slide holds the drawer level, so a travel longer than the drawer does not drop it |

### 2.3 `quarter_turn_latch` (approximation)

| Item | Model |
|---|---|
| Panel | static plate `max(1.6 x head, 40 mm)` square, 8 mm [assumed] |
| Head | hinge about the vertical, **opening counter-clockwise from above**, range `[0, turn_deg]` (the open stop) |
| `torque_nm` | no detent: the joint's dry friction over the whole turn; detent: 25 % running friction plus a 75 % torque toward locked over the first 8 deg (the template's motor) [assumed split]; breakaway = `torque_nm` |
| `head = wing` | a boss and a fin `head_diameter_mm` long, `head_height_mm` tall, 20 % of the diameter thick (at least 4 mm) |
| `head = slot` | a round head with a slot across it, 30 % of the diameter wide, half the height deep [assumed]: it needs a blade |
| Sites | `head` (top centre), `mark` (aux: the rim at local +x, shows the rotation) |
| State | `angle_deg` (counter-clockwise from locked), read within `[0, turn_deg]` |

### 2.4 `threaded_joint` (approximation)

| Item | Model |
|---|---|
| Seat | static base `max(1.6 x d, 40 mm)` square x 20 mm [assumed] and a stub with a `chamfer_mm` lead-in, 0.5 mm radial clearance to the bore [assumed] |
| Member | on two joints on the stub's axis: a slide (depth, downward) and a hinge (**tightening clockwise from above**, a right-hand thread); it starts resting on the stub top |
| Lead-in | the first `chamfer_mm + 3 mm` of depth is free of the thread [assumed]; the slide's friction loss (1.5 x the member's weight + 0.2 N) holds the member there until it is pushed |
| Thread | a fixed tendon `depth - pitch/(2 pi) x angle` limited above at the lead: past the thread start the member cannot go deeper than the helix allows (pushing does not advance it; tightening with an axial feed does, one pitch per turn). One-sided (the load flank): lifting it off is free |
| Seated | after `engage_turns` turns from the thread start: the slide's (near-hard) end stop |
| Torque ramp | engaged: resisting torque rises linearly from `start_torque_nm` to `final_torque_nm` when seated, and stays at `final_torque_nm` past it (a tightening step stops on torque); a Coulomb friction (S2c): it holds the member until the applied torque exceeds the ramp, then lets it slide at the ramp; the template's actuator is a torsion spring (100 N m/rad, damped 0.05 N m s/rad [assumed]) to an anchor that each control tick moves only as far as keeps the spring within the ramp; plus 0.02 N m of dry friction always [assumed] |
| Shapes | wall `max(1.5 mm, 15 % of d)` [assumed]: `coupling` 12-sided tube, `cap` the tube closed on top, `nut` a hexagon (`d` across corners) with a bore; contacts with the stub excluded |
| Sites | `member` (top centre), `seat` (stub top, static), `flat` (aux: a flat's centre) |
| State | `turns`, `angle_deg` (tightening from the start), `depth_mm` (from the stub top), read within the seat stop and the helical flank (`lead + pitch x turns`) |

### 2.5 `handwheel_valve` (approximation)

| Item | Model |
|---|---|
| Body | static valve body, bonnet and stem; wheel mid-plane **110 mm** above the table [assumed] |
| Wheel | hinge about the vertical, **opening counter-clockwise from above**, range `[0, travel_deg]`; hub `max(10 mm, 12 % of R)`; rim of 24 capsules `rim_section_mm` thick; `spokes` capsules 40 % of the section thick; `spokes = 0` a solid disc; density 2700 kg/m^3 [assumed] |
| Torque curve | `c(angle) x torque_nm`, c = 1.0 closed, 0.6 after 20 % of the travel, rising to 0.8 over the last 10 % [assumed shape]; half of `torque_nm` is dry friction, the rest the same Coulomb friction anchor as the thread's (S2c) |
| Sites | `wheel` (hub top), `spoke0` (aux: spoke 0 at the rim centre line) |
| State | `angle_deg` (counter-clockwise from closed), read within `[0, travel_deg]` |

### 2.6 `port_plug` (rigid)

| Item | Model |
|---|---|
| Socket | static, opening up: a floor (6 mm) and 16 convex wall pieces around a polygonal bore whose flats are `(plug_diameter + clearance)/2` from the axis, `depth_mm` deep, a 45 deg `chamfer_mm` lead-in (at most 80 % of the depth), walls at least 6 mm [assumed] |
| Plug | a free cylinder with a 0.6 mm tip chamfer [assumed], standing on the table 40 mm beside the socket along local +x [assumed start]; density 1400 kg/m^3 [assumed] |
| `insert_force_n` | three spring-loaded contact pads in the wall just below the chamfer (capsules on radial slides, preloaded 2 mm against an inner stop, squeezed 0.4 mm by a centred plug); their stiffness makes the pads' contact friction (the task friction, jittered) equal `insert_force_n`. A real, implicit contact friction (an explicit friction force chattered on a 6 to 20 g plug, so it was replaced) |
| `latch` | the plug gets a 0.8 mm groove where the pads sit when it is seated: they snap in and hold it [assumed geometry] |
| Sites | `plug` (top), `socket` (mouth centre, static), `plug_tip` (aux) |
| State | `depth_mm` (tip below the mouth while in the bore); within 0.1 mm of the floor it reads the full depth (`SEAT_TOL` [assumed]: the floor contact springs back a few hundredths of a millimetre when a press lets go) |

## 3. Primitives

All seven register on import (`rtt.studio.primitives.<primitive>`, names `studio_<primitive>`).
Each predicate ends with `criteria_verdict` (penetration above 1 mm is `physics_unreliable`,
a force or torque above the step's limit `constraint_exceeded`, an unmet criterion
`criterion_not_met`), then names the specific failure where it applies.

### 3.1 `grasp`, `suction`, `wrap` (the picks)

The v1 phases with the v1 arithmetic (`rtt.sim.controller`): transit, SETTLE 0.3 s at the
approach height, DESCEND 1.0 s to the tip height the head's v1 primitive plans, CLOSE 1.0 s plus
0.3 s squeeze, LIFT by `lift_mm` in 1.5 s, HOLD for `hold_s`. With a target bin: carry the part
over the bin (bottom 20 mm above its rim), lower it until its bottom is 12 mm above the floor,
open over 0.5 s, wait 0.4 s, retract 80 mm. Without a target: put it back and release; with
`hold_after`: end holding it. The step's primitive names the intended mechanism and sets its
tier; the physics is the tool's own.

Observed: `lift_mm` (peak rise during the step), `hold_s` (longest time lifted at least half the
planned lift, at least 10 mm, while touched), `in_target` (ends inside the bin's interior,
released), `tilt_deg` (at the end), `force_n` (peak sum of normal forces), `penetration_mm`,
`engaged` (the v1 criterion at its registration moment; informative). A part lifted and lost
before the release is `dropped`.

### 3.2 `hook_pull`

Engage the `handle` across its feature (across a horizontal bar or lip, along the face for a
door's vertical bar, across a knob's neck), fingers pre-shaped to it, `approach = top` (descend
onto it) or `front` (descend 30 mm out, move in), guarded; close with the head's primitive (1.0 s
plus 0.3 s); carry the engagement point along the arc about the observed hinge or along the slide
to the goal (the criterion plus 8 deg or 15 mm; a lid that must stay open goes past vertical, to
at least 100 deg, when its stop allows) at `speed_mm_s` (min-jerk); a door's heading turns with
it. A `force_n` at_least (S2d) is a goal on top of the motion's: once the pull reaches its goal it
goes on along the motion, toward the object's end and into it by the pull limit over the wrist
servo's stiffness plus 2 mm (80 % of `max_force_n`, else 30 N [assumed]; not when the step bounds
the motion on both sides), while the measured force is under that limit, until the force goal is
met, within the equal budget `force_press` plus that travel; then it lets go. Release to the
pre-shape (so no finger shoves the panel), back off 40 mm from the face, rise 60 mm.

**A lid's lip** (S2b) has nothing to grip or pinch: the tool's calibrated sideways hook is turned
toward the hinge, lowered outside the free edge (10 mm out, plus whatever of the tool reaches
further toward the lid than the hook), moved in under it, guarded, by the hook's depth less 3 mm
(at most the lip less 2 mm and 40 mm), raised 3 mm onto the underside and carried along the arc,
open. The soft arms' horizontal arms are such hooks (131, 151 and 171 mm); grippers, hands, cups
and the designed sliding hook gripper (whose hooks point inward, 4 mm) have none, reach nothing
under the lip and fail honestly.

Observed: the template state after the release, `slip_mm` (largest displacement of the handle
relative to the tool, in the tool frame, or the panel's for a hooked lip, engaged to release,
counted while the tool touches the object or the object lags more than 1 deg or 2 mm behind where
the tool is taking it: a lid that falls open past vertical ahead of a hook has not slipped from
it), `force_n` (peak net force), `penetration_mm`. Failing with more than 10 mm of slip is `slip`.

### 3.3 `twist`

Engagement from the observed axis and rotation mark: fingers across a latch's fin (or a slot
head, which they hold by friction only), across a member's flats (tips 3 mm above its bottom),
or, on a spoked handwheel, **spoke engagement**: the open (or slightly closed) fingers go down
into the gaps at the heading, from a 1 deg search over the calibrated footprint, that keeps every
tip clear of the hub, the rim and the spokes and puts a finger closest behind a spoke in the turn
direction; a solid disc is gripped across its rim if the fingers open wide enough, else pressed
on its hub. A cup seals on the top face (pressed 4 mm), a soft arm curls. Fingers are lowered
(S2b) so that their lowest point **while closing** sits at the planned tip height: a linkage
finger's tip swings down as much as 14 mm as it closes, and landing on the latch panel cost
42 N. Then the roll turns in `direction` at `speed_deg_s` (ramped over 0.2 s), holding while the
roll torque is at the limit (90 % of `max_torque_nm`, else 95 % of the wrist's) or, under a step
`max_force_n`, while the tool's contact force on the object is over 90 % of it; while it presses
down harder than that the wrist eases up (10 mm/s, at most 4 mm) [assumed]. Its goals (S2d) are
the rotation goal (the largest `angle_deg`, `turns` or `depth_mm` at_least, plus 3 deg) and a
`torque_nm` at_least; it turns until **both** are met (without either, the full travel, or to
seated for a thread; with only a torque goal it may turn as far as the object goes). Limits end it
first: an `angle_deg` or `turns` at_most in the working sense (3 deg before it), a `torque_nm`
at_most (the roll torque held under 95 % of it), a stall (0.3 s at a limit without progress), the
tool 90 deg past the object (past the rotation goal, or with a torque goal, ahead of the object:
slipping), the budget. A thread is fed: the wrist first takes the member back down to where it
was before the fingers closed (closing fingers can draw it up its lead-in, off the thread), then
descends one pitch per turn plus 1.5 mm of push. Then the roll target
backs off to the measured roll; fingers in a wheel's spoke gaps turn back by their free play, to
the middle of the gaps (so none lifts a spoke on the way out); the wrist holds where it is, lifts
3 mm while the head opens, and rises.

Observed: the template state at the end, `torque_nm` (peak roll torque), `slip_deg` (largest
difference between tool and object rotation from the moment the object first moves 0.5 deg until
the turn stops; when it never moves, the tool's whole rotation), `force_n`, `penetration_mm`.
Failing with more than 15 deg of slip is `slip`.

### 3.4 `insert`

Hold the member: pick a plug off the table; grip a threaded member in place across its flats, or
press its top with a cup or soft arm. Fingers grip a plug (S2b) low: 5 mm above the socket depth
(so they clear the socket when it is home), but with their first contact (3 mm above the tips
[assumed]) below 80 % of the plug's tip-over height `r / mu` (closing fingers that touch a
standing pin above that tip it over instead of sliding it; an 8 mm pin at mu 0.3 tips above
13 mm); and between the pads wherever they are mounted (the fin-ray gripper's pads span 0.6 to
26 mm along its x, off the TCP: it grips the plug at their middle). Align the plug's observed tip
over the mouth (`standoff_mm` above), then re-aim once it has settled 0.3 s (at most 15 mm). Push:
the z target descends at `speed_mm_s` but never more than `F / k_z` below the wrist (F: 80 % of
`max_force_n`, else 30 N [assumed]); under a step force limit it also rises while the measured
push is over 90 % of the limit (a tool's weight or a stiff cup adds to the servo); the x/y target
yields to the lateral force (0.002 m per N s, at most 3 mm) [assumed]; a plug stalled on the
mouth spirals (0.4 mm per turn up to 3 mm). A grip whose fingertips reach takes the plug home
(they stop 3 mm over the socket); a lower grip only starts it, 1 mm past the chamfer, where the
bore holds it upright. A threaded member goes to the criterion plus 0.5 mm (else the thread
start). **Press home** for a plug started, or stalled at least 1 mm in: let go (under a step force
limit the push eases off first: an opening finger on a pushed plug presses it down), rise until
the tip is 15 mm over the plug's top, close into a fist and press it home with the same force
limit. A `force_n` at_least (S2d) is a goal on top of the depth: at the goal depth the push goes on
pressing (the z servo's `F / k_z` reach) until the measured net force is over it, within the
equal budget `force_press`, on the first push or the press home. Release: the z target comes back
to the wrist (the push stops) before the fingers open; a
fist rises closed, and if the plug comes up with it (fingers that closed around its top) it holds
and opens first. Fingers are lowered for their lowest closing point, as in `twist`.

Observed: `depth_mm` at the end, `lateral_force_n` (peak force across the axis during the push),
`force_n`, `jam` (1 when it ends more than 0.5 mm short of the criterion, else of the full
depth), `penetration_mm`. Failing with `jam = 1` is `jam`.

### 3.5 `push`

Fingers in a fist (closure 1), a cup's vacuum off, a soft arm slack; the contact point is the
tool's tip moved to the side of its footprint that faces the push. A panel is pushed on its
outer face 15 mm in from the free edge along the arc to the goal (3 deg past it); a lid open past
vertical that must close is pushed only to 80 deg, its weight closes it, and the tool then
presses the closed lid home from above. A drawer is pushed on the front of its handle along the
slide (3 mm past the goal); a part is pressed down on its top until the press force (10 N, or 60 %
of the limit; a `force_n` at_least above that, S2d) for 0.5 s; a plug in its socket pressed down
to the goal depth. The push advances at `speed_mm_s` while the tool's net force is under the limit
(80 % of `max_force_n`, else 30 N [assumed]) and holds while it is over. A push to a criterion on
the object's state (S2d) ends when the **observed** state meets it (1 deg, 1 mm or 0.2 mm inside
its bounds [assumed]), not when the command arrives: the wrist servo lags behind the command under
the object's resistance (a drawer pushed shut stopped 7.7 mm open against a 5 mm criterion), so
the push goes on, up to the limit over the servo's stiffness plus 2 mm past its path. A
`force_n` at_least is a goal of the last push on top of its travel: it presses on, within that
reach and while the observed state still meets its criteria, until the force is reached and held
0.5 s, within the equal budget `force_press` plus that travel. Then hold 0.3 s, back off 30 mm,
rise 60 mm.

Observed: the template state at the end, `force_n` (peak net force), `penetration_mm`.

## 4. Equal budgets

The closed-loop phases have fixed time budgets, the same for every candidate
(`_common.BUDGET_S`). A step's planned duration includes them whether or not they are used, so
cycle times stay comparable.

| Budget | Seconds | Where |
|---|---|---|
| `engage_slack` | 0.4 | guarded descent onto a feature (on top of 1.0 s) |
| `hook_slack` | 0.5 | after a hook pull |
| `twist_slack` | 1.0 | on top of 1.3 x the nominal turn time |
| `insert_slack` | 1.0 | on top of 1.3 x the nominal insertion time |
| `insert_search` | 2.0 | spiral search at a socket mouth |
| `insert_reseat` | 2.6 | jam recovery (let go, press home), plus the press itself: 1.3 x the full depth less 1 mm at `speed_mm_s` (S2b; the same for every tool on the port, since a slipping grip can stall anywhere) |
| `push_slack` | 1.5 | force-limited pushing may pause |
| `force_press` | 1.0 | a push, an insert or a pull pressing on to a `force_n` at_least (S2d; only in a step with one, and for a part press only when it asks for more than the press force), plus 1.3 x the extra travel at the step's speed |

Every step is capped at 30 s (`MAX_STEP_S`).

## 5. Fidelity and caveats

A step's tier is the worst of its primitive's, its templates' and its effector's (STUDIO.md 6),
and every approximation outcome carries its caveats (the runner adds them; a step outcome
without them cannot be built). The texts are the contract's:

* latch: "Latches are an approximation: a detent torque on a revolute joint; no cam or spring
  geometry."
* threaded joint: "Threads are an approximation: a helical joint with a torque ramp; no thread
  flanks, cross-threading or galling. A screen, not a promise." (S2's helix is a one-sided
  tendon limit, section 2.4.)
* handwheel valve: "Valves are an approximation: a resistive torque curve on a revolute joint;
  no packing or seat dynamics."
* suction primitive: "Suction is an approximation: the seal is a geometric test gating a capped
  adhesion force; no air flow, leak rate or porosity." (plus the suction family's own caveat)
* wrap primitive: "The soft arm is an approximation: rigid segments on spring joints curled by a
  cable." (plus the soft-arm family's own caveat)

Lids, doors, drawers and ports are `rigid` per the contract; their catch, friction and pad
models are named [assumed] choices above.

## 6. Gate results

Design gate S2 (STUDIO.md 7.2), measured with `tests/test_studio_primitives.py` cases over 10
seeds (the v1 jitter: +-0.5 mm, +-10 % friction) and 4 placement errors (+-2 mm in x and y, hidden
from the controller): 14 conditions per effector. Rerun after S2b (2026-10-18) with the S2b
cases (the hero-task limits, section 8) added below the line.

| Primitive | Task | Reference effector | Success | Wrong effector | Success | Max penetration on success | Median step time |
|---|---|---|---|---|---|---|---|
| grasp | steel slug into a tote | two-finger adaptive, pinch | 14/14 | rigid-pad gripper A (4.1 N grip) | 0/14 (criterion not met) | 0.32 mm | 11.4 s |
| suction | 600 g flat item into a tote | 25 mm cup | 14/14 | 10 mm cup (4.7 N hold) | 0/14 (criterion not met) | 0.08 mm | 11.6 s |
| wrap | lift a flanged bushing | 16-segment soft arm | 14/14 | 16 mm cup (bore) | 0/14 (criterion not met) | 0.28 mm | 10.4 s |
| hook_pull | drawer by its bar, 120 mm | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (slip) | 0.22 mm | 11.0 s |
| twist | handwheel a quarter turn (3 N m) | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (slip) | 0.20 mm | 9.5 s |
| insert | 20 mm plug, 0.4 mm clearance, 2 mm chamfer | two-finger adaptive, power | 13/14 | 12-segment soft arm | 0/14 (jam) | 0.57 mm | 18.2 s |
| push | press a box down (5 N) | two-finger adaptive, pinch | 14/14 | 12-segment soft arm | 0/14 (criterion not met) | 0.06 mm | 5.3 s |
| insert (S2b) | 8 mm plug, 30 mm, 0.2 mm, 1 mm chamfer, to 15 mm at 20 N | fin-ray pad gripper | 14/14 | two-finger adaptive, power | 0/14 (physics unreliable) | 0.81 mm | 18.5 s |
| insert (S2b) | the same, rigid-pad gripper A | rigid-pad gripper A | 11/14 (jam: its 4.1 N grip lets the plug slide) | | | 0.62 mm | 18.1 s |
| insert (S2b) | the same port, 12 mm plug | rigid-pad gripper A | 13/14 | two-finger adaptive, power | 0/14 (jam) | 0.38 mm | 18.2 s |
| hook_pull (S2b) | free 380 x 280 lid by its lip to 90-110 deg at 20 N | 12-segment soft arm | 14/14 | two-finger adaptive, pinch | 0/14 (slip) | 0.44 mm | 24.6 s |
| hook_pull (S2b) | 400 x 300 door by its handle to 80 deg at 30 N, off the home pose | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (constraint exceeded) | 0.26 mm | 18.2 s |

The insert step times grew by 1.8 s (16.4 s in S2): every tool on a port now carries the same
press-home budget (section 4).

No MuJoCo reset (`physics_invalid`) in any gate run. Two in-process runs and a spawned worker
give the same `traj_sha` for every gate task.

**Every template variant** (seed 0, the two-finger adaptive gripper; `power` for the plug
inserts):

| Variant | Step | Result | Measured |
|---|---|---|---|
| lid, lip | hook pull to 80 deg | fails (slip) | a straight finger cannot get under a horizontal lip; the 12-segment soft arm hooks under it: 110 deg, slip 6.2 mm (S2b) |
| lid, handle | hook pull to 80 deg | passes | 110 deg (on its stop), slip 11.7 mm |
| door, lip | hook pull to 60 deg | passes | 68 deg, slip 0.4 mm (edge pinch) |
| door, handle | hook pull to 60 deg | passes | 68 deg, slip 0.1 mm |
| lid open at 108 deg | push closed (40 N limit) | passes | 0 deg, 8.4 N |
| door open at 70 deg | push closed (40 N limit) | passes | 0 deg, 32 N |
| drawer, bar / knob / lip | hook pull to 120 mm | passes | 160 / 132 / 120 mm; slip 7.3 / 0.5 / 33.8 mm |
| drawer out 150 mm | push closed | passes | 4.9 mm, 5.3 N |
| latch, wing, 0.8 N m (default) | twist 90 deg | fails (slip) | 0.69 N m peak: the fin slips in the fingers (stiffer linkage grippers turn it, section 8) |
| latch, wing, 0.3 N m | twist 90 deg | passes | 88.5 deg, slip 0.1 deg |
| latch, slot, 0.3 N m | twist 90 deg | fails (slip) | friction on a round head (needs a blade) |
| coupling (default) | insert to the thread start | passes | 4.0 mm |
| coupling, 5 N m final (default) | twist 2.5 turns | fails (slip) | 1.6 turns, 0.24 N m: a round member slips |
| nut / cap, 0.3 N m final | twist 2.5 turns | passes | 2.51 turns, slip 0.5 / 0.8 deg |
| valve, 3 spokes, 3 N m (default) | twist 90 deg | passes | 88.6 deg, 2.9 N m, slip 0.1 deg |
| valve, solid disc, default | twist 90 deg | fails (slip) | too wide to grip: pressed on the hub |
| valve, 50 mm disc, 1 N m | twist 90 deg | fails (slip) | gripped across the rim: 0.27 N m by friction |
| port, default 12 mm | insert | fails (jam) | 2.2 mm: the adaptive fingers curl into a V around the slim plug; rigid-pad gripper A seats it, 15.0 mm (S2b) |
| port, 20 mm (with or without latch) | insert | passes | seated, 15.0 mm |
| port, 20 mm plug | grasp, lift | passes | 100 mm |
| threaded member | grasp, lift | fails | it lives on its stub (section 7) |

## 7. Known limits

* **A threaded member lives on its stub.** It cannot be picked off the table: grasp lifts it
  only along the axis (not at all past its start), insert pushes it down the lead-in to the
  thread start and jams beyond, twist runs it down. A use case that picks a loose coupling and
  starts it on a stub needs the member to start free (not modelled: the helical coupling is a
  constraint between joints on the stub's axis).
* **Torque capacity is the effector's.** A round member is turned by friction only: two jaws
  give `2 mu N r`, so the 40 mm coupling (r 20 mm, mu 0.3) needs about 170 N per jaw for its
  2 N m; the best screened tool (an angular-jaw gripper) peaks at 1.2 to 1.3 N m and slips
  (`slip`). A wing is turned by
  the pads pressing its faces, which the fin pushes apart: a 0.8 N m detent on a 25 mm wing needs
  about 64 N of finger holding force, which the stiff linkage grippers have (88.5 deg within 5 N)
  and the adaptive and most parallel grippers do not (0.69 N m, `slip`). A handwheel turns at its
  3 N m because the fingers push the spokes; with only one finger on a spoke the wheel takes the
  whole push sideways (2.9 N m at 50 mm is 58 N), so a two-finger gripper on a three-spoke wheel
  exceeds a 10 N force limit honestly; three fingers 120 deg apart share it (5 to 7 N).
* **A lid's lip** is hooked from under its free edge (S2b), which needs a hook (the soft arms'
  horizontal arms) and room under the edge: a lid flush with, or inset in, its bin's walls cannot
  be opened by any top-down tool (section 8); a straight-fingered tool never can.
* **Small plugs.** Two-finger adaptive grippers curl into a V around an 8 or 12 mm plug and
  squeeze it out (`jam`, or `physics_unreliable` when the pin is flung); parallel pads seat
  them (S2b). The catalog hands' pinch closes away from their TCP and knocks a slim standing
  plug over on the descent; a hand-specific pinch pose is not modelled.
* **The home pose is the table centre.** Every segment starts with the wrist at
  `HOME_HEIGHT` over the table centre (the contract's segment start). An object there that rises
  into the tool's reach (a 410 mm door; a lid under a 244 mm fin-ray gripper) has the tool start
  inside it: 9 to 59 mm of penetration before any controller runs. Not a controller fault
  (contract change request in section 8).
* **Directions.** Latches and valves open counter-clockwise, threads tighten clockwise; a twist
  step in the other direction meets the stop and fails.
* **Perception.** A valve's angle is observed modulo a turn; a thread's turns only past the
  thread start (from its depth).

## 8. S2b: the hero-task limits

S5's search ran the hero tasks and found physics limits that looked like controller quality.
Each was traced in a single run (per-tick contact forces and positions), fixed where the fault
was in a controller or a template, and kept where the tool or the plan cannot do the step. The
screens below run every S5 default candidate (presets, library entrants, designed effectors) on
one step of S5's re-recorded hero plans, in the first 3 of the report's conditions, after the
fix; "before" is S5's own screen of the same plans without S2b.

| Hero step | Before (S5) | After (S2b) | What it was |
|---|---|---|---|
| H1 `open_lid` (lip, 300 x 200 lid on a 300 x 200 tote) | 0/29 tools; slip about 305 mm, or 24 to 32 N against 20 N | 0/49 | the plan: the lid is flush with the tote's walls, so nothing can get under its edge (honest) |
| H1 `open_lid`, the same with a 15 mm lip overhang (330 x 230 lid) | not run | soft arms 12/16/20, 40 mm cup, button pusher 3/3 each (110 deg, 3 to 5 N) | the hook-under motion |
| H3 `unlock_latch` (0.8 N m wing, 10 N) | 0/30; 52 to 909 N | 2 tools 3/3 (88.5 deg, 3.0 and 4.8 N) | linkage tips landing on the panel; no force limit in the turn |
| H3 `open_door` | 11/29 | 11 tools 3/3 | unchanged (S4c moved the latch out of its footprint) |
| H3 `open_valve` (3 N m, 10 N) | 0/34; hook gripper 83.6 deg at 14.3 N | 2 tools 3/3, 1 at 2/3 (83.5 deg, 4.9 to 6.7 N) | a hooked finger lifting a spoke on the way out |
| H3 `insert_plug` (8 mm, 40 mm long, 13 to 17 mm, 20 N) | 0/25; jam, one at 6.3 mm penetration | 5 tools 3/3, 2 at 2/3 (seated, 13.7 to 16.0 N) | grip height, pad position, budget, push and release |
| H2 `insert_coupling` | 22/29 | 32 tools 3/3 | unchanged in substance |
| H2 `tighten_coupling` (2 N m on a round 40 mm coupling) | 0/29; best 1.33 N m, 32.6 deg slip | 1 tool 1/3 | friction capacity (honest, section 7) |

**1. Small-plug insert.** The 8 and 12 mm plugs jammed because the grasp tipped and cocked them.
Fixed in the controller: the fingers grip a standing plug below its tip-over height (`r / mu`:
13 mm for an 8 mm pin at mu 0.3), between the pads wherever they are mounted (the fin-ray pads
sit 0.6 to 26 mm off the TCP), from their lowest closing point (linkage fingers swing 14 mm
down); a grip too low to take the plug home starts it 1 mm past the chamfer and the fist presses
it home, inside one press budget for every tool on the port; the push stops before the fingers
let go, and eases off first under a step force limit; a fist that has pressed a plug home lets
go before rising if the plug comes up with it; the z servo backs off a measured push over 90 %
of a step's force limit. The clearance and chamfer are unchanged. A plug within 0.1 mm of the
socket floor reads seated (the floor contact springs back a few hundredths of a millimetre).
Before: 0/57 runs on the 8 mm plug (rigid-pad gripper A jammed at 2.2 mm). After, 30 mm plug to
15 mm at 20 N: fin-ray 14/14, rigid pads A 11/14 (its 4.1 N grip lets the plug slide in three),
12 mm rigid pads A 13/14. The 6.3 mm penetration S5 saw (air-actuated mini gripper) is now a
`jam` at 0.74 mm. Honest limits kept: adaptive two- and three-finger grippers curl into a V
around a slim pin and eject it; the catalog hands' pinch closes off their TCP and knocks the pin
over on the descent.

**2. Lid lip.** Every tool slid about 428 mm along the lip. The S2 plan pinched the edge from
the top, which no tool can do on a flat panel. Fixed: the hook-under motion (section 3.2) with
the tool's calibrated sideways hook, slip counted only while the tool touches the lid or the lid
lags (a lid falling open past vertical ahead of the hook is not slip), guarded moves that stop
on any object (the tote's wall), and states read within the joint limits (a lid on its 110 deg
stop read 110.0017 deg and failed "at most 110"). A free lid opens (12-segment soft arm 14/14,
110 deg, 3 N, slip 6 mm); a lid overhanging its tote by 10 mm opens with the 12- and 16-segment
arms, by 15 mm with all three. The designed sliding hook gripper's hooks point inward (4 mm):
it cannot reach under a lip from outside and fails honestly, as does the swinging hook gripper
(it now builds, CR-S5-2, and has no outward hook either). **H1's lid cannot open as
planned**: S4c put a 300 x 200 lid at its 300 x 200 tote's position, so the edge is flush with
the walls and rests on them; there is no gap to hook into for any tool (0/49).

**3. Door pull.** The door alone at the table centre still penetrated 16 mm, from the first
tick: the wrist starts every segment at its home pose over the table centre, inside a 410 mm
door. With the door moved off the home pose (y = 150 mm) it opens to 88 deg at 2.7 N with
0.23 mm (two-finger adaptive, 3/3; 11 tools pass on S4c's plan). Not a controller fault; no
change.

**Twist under a force limit** (S5 priority 2). The latch: linkage tips swing down as they close
and landed on the panel (one 42.8 N tick); the turn ignored the step's force limit. Fixed: the
fingers are lowered for their lowest closing point; the roll holds while the contact force is
over 90 % of `max_force_n`, and the wrist eases up while it presses down over it; it lifts 3 mm
while the head opens. The valve: the hook gripper's fingers sat behind the spokes after the turn
and lifted the wheel on the way out (14.3 N, upward); fixed by turning back by the free play
before rising. Kept: most grippers hold a 25 mm wing's 0.8 N m detent with too little finger
force, and a two-finger gripper on a three-spoke wheel pushes one spoke, which loads the wheel
sideways with the whole torque (section 7).

**Thread twist** (S5 priority 3): honest. The coupling is round to the fingers; its 2 N m needs
about 170 N per jaw at mu 0.3 and r = 20 mm. The grip is held closed through the turn (the
head's own v1 primitive, closure 1); no re-grip can add force.

### Contract change requests

* **CR-S2b-1 (Task Engineer, S4):** a hinged lid with `feature = lip` over a bin must overhang
  the bin's walls (the contract calls `feature_mm` the "lip overhang"): lid depth at least the
  bin's width plus twice `feature_mm`, and width at least its length plus twice `feature_mm`
  (the lid sits centred on the bin under the lid-on-bin rule), or use `feature = handle`.
  Measured: overhang 0 (H1 as planned) opens with no tool; 10 mm with the 12- and 16-segment soft
  arms; 15 mm with all three soft arms, a 40 mm cup and the button pusher. Alternative for the
  contract owners: the template models the lip as a `feature_mm` extension beyond `depth_mm` at
  the free edge, with S4c's lid footprint extended to match (its geometry test would otherwise
  fail).
* **CR-S2b-2 (runner/episode, or the Task Engineer's layout):** the segment start puts the wrist at
  `HOME_HEIGHT` over the table centre, so an object there that reaches into the tool's envelope
  has the tool start inside it (H3's door: 9 to 34 mm for every tool; H1's lid with the 244 mm
  fin-ray gripper or the long industrial hand: 20 to 59 mm). Either start each segment clear of
  every object (raise the home above the tallest object under it by the tool's length), or have
  the layout check keep the table centre's tool envelope free.
* **Criteria at a stop or a seat** (for S4, informational): S4c's window rule already removed
  the zero-wide `[4.0, 4.0]` window; S2b reads states within the joints' limits and a plug within
  0.1 mm of the floor as seated, so a criterion at the stop or the full depth is met by an
  object driven onto it.

## 9. S2c: tool stacks and the friction torques

**Heads on a changer or a bracket.** The 80 mm cobot hand turned H3's valve on the flange
(83.7 deg at 1.96 N m) and never touched it on a tool changer with a 15 to 40 mm stack (0 N,
0 deg, 170 deg slip). The vendored catalog grippers and hands level their housing at `MOUNT_Z`
whatever `mount_z` they are given, so on a changer they sat inside the stack at one height for
every stack, while `tool_of` placed their tip by the stack: on a 40 mm changer the planner put the
tip 28 mm lower than it was (a 5 mm stack happened to be close). The scene now builds every head
on a changer or a bracket adapter under a mount frame whose `MOUNT_Z` plane is the stack's tool
face (docs/STUDIO.md 4.2), so every family hangs the stack height lower and the planned tip moves
with it. Before, on a 40 mm changer: fin-ray grasp and wrap, rigid-pad insert, five-finger push
and a library parallel gripper's twist all failed without touching the part (push and twist on the
default 25 mm changer too); after, all 7 gate tasks with their references and these vendored tools
end on 25 and 40 mm changers as on the flange (`test_a_gate_task_on_a_tool_changer_ends_as_on_the_flange`).
A head on a bracket had the same error, 4 mm (adapter 8 mm against `MOUNT_Z` 12 mm).

**Friction torques.** The thread's ramp and the valve's curve were explicit motors,
`-T tanh(omega / omega_s)`: they hold nothing at rest and, with a light member, flipped sign every
tick: 1 N m against a 4.1 N m ramp turned the coupling 389 deg in 2 s, 3 N m 1348 deg, and a stiff
driver at 60 deg/s needed about 0 N m; the valve crept 86 deg in 1 s at 1.6 of its 1.8 N m. Both
are now a Coulomb friction (an anchored torsion spring, section 2.4): below the ramp the member
does not turn (0.001 deg/s after its elastic twist, under 1 deg per 1.7 N m), above it, it turns;
a driver needs the ramp within 10 %; runs stay deterministic (the anchor is the actuator's control,
in `MjData`). H2's tighten step is now a clean test of grip: the stock part jaw slips at its
friction capacity, 0.6 N m, 0.1 turn in (before: 1.19 N m peaks from the motor's spikes); the
fin-ray gripper at 0.36 N m.

## 10. S2d: every goal of a step

**The bug.** A twist stopped at the first goal it reached. H2's tighten step asks for `torque_nm`
3 to 8 N m and `turns` at least 0.25; every tool that grips the coupling without slipping stopped
at 0.26 turns, on the start of the thread's ramp, at 0.14 to 1.25 N m, and failed
`criterion_not_met`. A twist now turns until every at_least goal is met (section 1): the rotation
and the torque, the torque counted once it has been 3 % over its goal for 0.03 s (a stop on the
first tick over 3.0 N m left the predicate's check-tick peak just under it). Limits still end it:
a `torque_nm` at_most holds the roll torque under 95 % of it, an `angle_deg` or `turns` at_most
in the working sense stops it 3 deg short, and with a torque goal slipping is the tool running
90 deg ahead of the object. Two side fixes surfaced on the way: the thread twist first takes the
member back down to its depth before the fingers closed (the designed four-finger hand drew the
coupling 4 mm up its lead-in as it closed and then spun it off the thread, at 0.06 mm depth), and
several criteria on one quantity now all hold (the largest at_least, the smallest at_most;
before, the last one listed won).

H2's tighten step from the screen's starting state (the screen's context tool inserts the
coupling first), three conditions (nominal and the screen's two placement errors):

| Tool | Before | After |
|---|---|---|
| library angular-jaw gripper | 0/3, criterion not met: 0.26 turns, 1.23 to 1.25 N m, slip 2.0 to 2.8 deg | 0/3, slip: 0.46 to 0.51 turns, 1.31 to 1.36 N m (its grip's capacity) |
| library two-finger linkage gripper | 0/3, criterion not met: 0.26 turns, 1.20 N m, slip 0.7 deg | **3/3**: 0.75 turns, 3.10 to 3.11 N m, slip 2.4 deg |
| designed four-finger hand (b56a) | 0/3, criterion not met: 0.26 turns, 0.14 N m, depth 0.06 mm (lifted off the thread) | 0/3: slip x2 and physics unreliable x1, 2.28 to 2.36 turns, 0.64 to 0.70 N m, depth 3.7 to 4.4 mm |
| designed three-digit tentacle (f293) | 0/3, criterion not met: 0.26 turns, 0.45 to 0.72 N m | 0/3, criterion not met: 2.47 turns, 0.44 to 0.69 N m, depth 0.9 to 1.1 mm (the soft arm draws the member up while it turns) |

So the tools that grip without slipping now meet the step when their grip holds 3 N m (the
linkage gripper), and fail honestly at their own capacity when it does not. The angular jaw's end
depth (0.5 to 2.8 mm, from 4.4) is its release dragging the member up after the slip; the thread
holds a member only against going deeper.

**hook_pull, insert, push.** The same rule: their extra at_least goal is `force_n` (the other
measured quantities are limits). Before, each stopped at its travel or depth goal (or, for a part
press, at the 10 N press) whatever the force goal; now each presses or pulls on to it, within
the force limit and the equal `force_press` budget (section 4), and a push to a criterion on the
object's state ends when the observed state meets it. Probe, one seed, two-finger adaptive
gripper (pinch; power for the plug):

| Step | Before | After |
|---|---|---|
| press a box, `force_n` >= 20, 40 N limit | criterion not met, 10.1 N | success, 20.8 N |
| press a box, `force_n` >= 5 (the gate case) | success, 10.1 N | success, 10.1 N, same step time |
| push a drawer shut, `travel_mm` <= 5 | criterion not met, 7.7 mm (the servo lags the command under the drawer's friction) | success, 3.1 mm |
| push a drawer shut, `travel_mm` <= 5 and `force_n` >= 15 | criterion not met, 7.7 mm, 5.3 N | criterion not met: 0.0 mm, 13.2 N (pressed into the stop; the adaptive fingers slide off the handle) |
| pull a drawer 120 mm, `force_n` >= 25 | criterion not met, 160 mm, 11.6 N | criterion not met: pulled on to its end, 246 mm, 15.2 N (the pinch grip holds no more) |
| insert a 20 mm plug to 14 mm, `force_n` >= 20 | success, 46 N (seating) | success, the same run |
| insert the plug, `force_n` >= 60, 100 N limit | criterion not met, 32.9 N | criterion not met, 38.9 N (pressed on; the grip's friction caps it) |
| latch to 85 deg and `torque_nm` >= 0.3 | criterion not met: stopped at the breakaway torque, 0 deg | success, 90 deg, 0.33 N m |

Regression tests: `test_a_twist_with_turns_and_torque_goals_turns_until_both_are_met` (a nut on
a 0.05 to 0.3 N m ramp, a quarter turn and 0.2 N m: before, 0.26 turns and 0.15 N m; now
2.25 turns and 0.21 N m; 0.5 N m, over the pinch grip's 0.35 N m, ends as slip after 1.6 turns)
and `test_a_push_meets_its_force_goal_and_its_observed_travel_goal`. Both fail on the code before
S2d.

**Gate.** The S2 gate (section 6, 14 conditions per effector) is unchanged by S2d: grasp,
suction, wrap, hook_pull, twist and push 14/14 with their references, insert 13/14, every wrong
effector 0/14 with the same failure modes, the same penetrations and step times, no MuJoCo reset.
The twist variants (latch, coupling, nut, cap, valve and disc cases with the pinch gripper) end
exactly as before.

**Left open.** Pushing a lid shut from past vertical (opened to 110 deg, then `angle_deg` <= 3)
exceeds a 40 N limit, 43.9 N, before and after S2d: the tool already presses 43.7 N on the lid
when its press from above starts, so the load comes from the move down to the press's standoff
over the lid that gravity closed, which is position-controlled. The engineer's
check that a `torque_nm` at_least stops a twist on a latch or valve before it turns
(`rtt.studio.engineer.checks`, not an S2 file) still holds for a twist with only a torque goal; a
twist with a rotation goal as well now turns to it.
