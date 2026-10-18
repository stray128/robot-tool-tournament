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
| Torque ramp | engaged: resisting torque rises linearly from `start_torque_nm` to `final_torque_nm` when seated, and stays at `final_torque_nm` past it (a tightening step stops on torque); the template's motor as a velocity-saturated friction (0.3 rad/s) plus 0.02 N m of dry friction always [assumed] |
| Shapes | wall `max(1.5 mm, 15 % of d)` [assumed]: `coupling` 12-sided tube, `cap` the tube closed on top, `nut` a hexagon (`d` across corners) with a bore; contacts with the stub excluded |
| Sites | `member` (top centre), `seat` (stub top, static), `flat` (aux: a flat's centre) |
| State | `turns`, `angle_deg` (tightening from the start), `depth_mm` (from the stub top), read within the seat stop and the helical flank (`lead + pitch x turns`) |

### 2.5 `handwheel_valve` (approximation)

| Item | Model |
|---|---|
| Body | static valve body, bonnet and stem; wheel mid-plane **110 mm** above the table [assumed] |
| Wheel | hinge about the vertical, **opening counter-clockwise from above**, range `[0, travel_deg]`; hub `max(10 mm, 12 % of R)`; rim of 24 capsules `rim_section_mm` thick; `spokes` capsules 40 % of the section thick; `spokes = 0` a solid disc; density 2700 kg/m^3 [assumed] |
| Torque curve | `c(angle) x torque_nm`, c = 1.0 closed, 0.6 after 20 % of the travel, rising to 0.8 over the last 10 % [assumed shape]; half of `torque_nm` is dry friction, the rest the template's motor (velocity saturated at 0.1 rad/s) |
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
it. Release to the pre-shape (so no finger shoves the panel), back off 40 mm from the face, rise
60 mm.

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
on its hub. A cup seals on the top face (pressed 4 mm), a soft arm curls. Then the roll turns in
`direction` at `speed_deg_s` (ramped over 0.2 s), holding while the roll torque is at the limit
(90 % of `max_torque_nm`, else 95 % of the wrist's); it stops at the goal (the criterion plus 3
deg; else the full travel, or to seated for a thread), at a `torque_nm` criterion's torque, on a
stall (0.3 s at the limit without progress) or 90 deg past the goal. A thread is fed: the wrist
descends one pitch per turn plus 1.5 mm of push. Then the roll target backs off to the measured
roll, the head opens and the wrist rises.

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
start). **Press home** for a plug started, or stalled at least 1 mm in: let go, rise until the tip
is 15 mm over the plug's top, close into a fist and press it home with the same force limit.
Release: the z target comes back to the wrist (the push stops) before the fingers open.

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
of the limit) for 0.5 s; a plug in its socket pressed down to the goal depth. The push advances
at `speed_mm_s` while the tool's net force is under the limit (80 % of `max_force_n`, else
30 N [assumed]) and holds while it is over. Then hold 0.3 s, back off 30 mm, rise 60 mm.

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
from the controller): 14 conditions per effector.

| Primitive | Task | Reference effector | Success | Wrong effector | Success | Max penetration on success | Median step time |
|---|---|---|---|---|---|---|---|
| grasp | steel slug into a tote | two-finger adaptive, pinch | 14/14 | rigid-pad gripper A (4.1 N grip) | 0/14 (criterion not met) | 0.32 mm | 11.4 s |
| suction | 600 g flat item into a tote | 25 mm cup | 14/14 | 10 mm cup (4.7 N hold) | 0/14 (criterion not met) | 0.08 mm | 11.6 s |
| wrap | lift a flanged bushing | 16-segment soft arm | 14/14 | 16 mm cup (bore) | 0/14 (criterion not met) | 0.28 mm | 10.4 s |
| hook_pull | drawer by its bar, 120 mm | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (slip) | 0.22 mm | 11.0 s |
| twist | handwheel a quarter turn (3 N m) | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (slip) | 0.18 mm | 9.5 s |
| insert | 20 mm plug, 0.4 mm clearance, 2 mm chamfer | two-finger adaptive, power | 13/14 | 12-segment soft arm | 0/14 (jam) | 0.56 mm | 16.4 s |
| push | press a box down (5 N) | two-finger adaptive, pinch | 14/14 | 12-segment soft arm | 0/14 (criterion not met) | 0.06 mm | 5.3 s |

No MuJoCo reset (`physics_invalid`) in any gate run. Two in-process runs and a spawned worker
give the same `traj_sha` for every gate task.

**Every template variant** (seed 0, the two-finger adaptive gripper; `power` for the plug
inserts):

| Variant | Step | Result | Measured |
|---|---|---|---|
| lid, lip | hook pull to 80 deg | fails (slip) | a straight finger cannot get under a horizontal lip |
| lid, handle | hook pull to 80 deg | passes | 110 deg (on its stop), slip 11.7 mm |
| door, lip | hook pull to 60 deg | passes | 68 deg, slip 0.4 mm (edge pinch) |
| door, handle | hook pull to 60 deg | passes | 68 deg, slip 0.1 mm |
| lid open at 108 deg | push closed (40 N limit) | passes | 0 deg, 8.4 N |
| door open at 70 deg | push closed (40 N limit) | passes | 0 deg, 32 N |
| drawer, bar / knob / lip | hook pull to 120 mm | passes | 160 / 132 / 120 mm; slip 7.3 / 0.5 / 33.8 mm |
| drawer out 150 mm | push closed | passes | 4.9 mm, 5.3 N |
| latch, wing, 0.8 N m (default) | twist 90 deg | fails (slip) | 0.69 N m peak: the fin slips in the fingers |
| latch, wing, 0.3 N m | twist 90 deg | passes | 88.5 deg, slip 0.1 deg |
| latch, slot, 0.3 N m | twist 90 deg | fails (slip) | friction on a round head (needs a blade) |
| coupling (default) | insert to the thread start | passes | 4.0 mm |
| coupling, 5 N m final (default) | twist 2.5 turns | fails (slip) | 1.6 turns, 0.24 N m: a round member slips |
| nut / cap, 0.3 N m final | twist 2.5 turns | passes | 2.51 turns, slip 0.5 / 0.8 deg |
| valve, 3 spokes, 3 N m (default) | twist 90 deg | passes | 88.6 deg, 2.9 N m, slip 0.1 deg |
| valve, solid disc, default | twist 90 deg | fails (slip) | too wide to grip: pressed on the hub |
| valve, 50 mm disc, 1 N m | twist 90 deg | fails (slip) | gripped across the rim: 0.27 N m by friction |
| port, default 12 mm | insert | fails (jam) | 2.0 mm: the slim plug tilts in the grasp |
| port, 20 mm (with or without latch) | insert | passes | seated, 15.0 mm |
| port, 20 mm plug | grasp, lift | passes | 100 mm |
| threaded member | grasp, lift | fails | it lives on its stub (section 7) |

## 7. Known limits

* **A threaded member lives on its stub.** It cannot be picked off the table: grasp lifts it
  only along the axis (not at all past its start), insert pushes it down the lead-in to the
  thread start and jams beyond, twist runs it down. A use case that picks a loose coupling and
  starts it on a stub needs the member to start free (not modelled: the helical coupling is a
  constraint between joints on the stub's axis).
* **Torque capacity is the effector's.** Two-finger grippers hold a round member by friction
  (0.2 N m or so at mu 0.3); the default latch (0.8 N m) and thread (5 N m final) exceed what a
  two-finger adaptive gripper transmits through a fin or a coupling, so those steps fail honestly
  (`slip`) with catalog effectors; a handwheel turns at its 3 N m because the fingers push the
  spokes.
* **A lid's lip** cannot be opened by a top-down straight-fingered tool (no hook under it, no
  pinch of a horizontal edge); a lid needs a handle, or a hook effector.
* **Small plugs.** The default 12 mm plug with 0.2 mm clearance and a 1 mm chamfer tilts in the
  two-finger grasp and jams; 20 mm with 0.4 mm and 2 mm passes.
* **Directions.** Latches and valves open counter-clockwise, threads tighten clockwise; a twist
  step in the other direction meets the stop and fails.
* **Perception.** A valve's angle is observed modulo a turn; a thread's turns only past the
  thread start (from its depth).
