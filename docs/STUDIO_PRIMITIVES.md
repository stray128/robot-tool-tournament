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
11. [S2e: a push's approach](#11-s2e-a-pushs-approach)
12. [Studio v2: the wrist pitch in the primitives](#12-studio-v2-the-wrist-pitch-in-the-primitives)
13. [Studio v2: four job templates](#13-studio-v2-four-job-templates)
14. [Studio v2: a push that falls behind, and reach for every primitive](#14-studio-v2-a-push-that-falls-behind-and-reach-for-every-primitive)

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
* **Wrist.** Every controller commands all six wrist servos (x, y, z, yaw, roll, pitch). With the
  tool straight down (pitch 0) the palm yaw and the continuous roll turn about the same vertical
  axis; the tool tilts only where it matters (section 12). Controllers
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
Each predicate ends with `criteria_verdict` (penetration: the runner measures it over every
contact pair except the designed ones, the tool's own links and a port's pads, STUDIO.md 3.3; a
pair held more than 1 mm deep for more than 10 ms is `physics_unreliable`, a shorter excursion is
flagged `penetration_transient` and not failed; a force or torque above the step's limit
`constraint_exceeded`, an unmet criterion `criterion_not_met`), then names the specific failure
where it applies.

**Letting go** (S2f, studio v2). A part is set down before it is let go, its lowest point at most
`RELEASE_HEIGHT_MAX_MM` = 20 mm [assumed] above what it comes to rest on (`_common`); a lid is
carried to its stop rather than dropped onto it (sections 3.2 and 3.5).

### 3.1 `grasp`, `suction`, `wrap` (the picks)

The v1 phases with the v1 arithmetic (`rtt.sim.controller`): transit, SETTLE 0.3 s at the
approach height, DESCEND 1.0 s to the tip height the head's v1 primitive plans, CLOSE 1.0 s plus
0.3 s squeeze, LIFT by `lift_mm` in 1.5 s, HOLD for `hold_s`. With a target bin: carry the part
over the bin (bottom 20 mm above its rim) and set it down on the floor, or on whatever already
lies in the bin under it (its top seen as the transit height is seen); without a target: set it
back down where it was; then open over 0.5 s, wait 0.4 s, retract 80 mm. With `hold_after`: end
holding it. The step's primitive names the intended mechanism and sets its tier; the physics is
the tool's own. A pick whose grasp, or whose set-down in its target bin, would put the palm outside
its travel (+-350 mm less 5 mm) is not attempted: it fails `out_of_reach` (section 14).

**Setting down** (S2f): the part is lowered (at the transit's 0.4 m/s peak) until its bottom is 15 mm
above the support as planned, then slowly (16 mm/s mean, 30 mm/s peak [assumed]) on toward 5 mm
below it; both moves are guarded on the wrist's force sensor (every contact of the tool with the
world): they stop advancing once the vertical load the tool carries has changed by 3 N [assumed]
from what it carried when the descent began, that is once the support has taken the part (a part
that hangs lower than planned lands early and stops there). The descent runs its whole planned
time: a guard that holds it back for a moment lets it go on once the load is off again (ending it
at the first trip let a 4 kg item on a 40 mm cup go 9 mm up, to fall 9.6 mm at 0.36 m/s). Before it
opens, the controller compares where the part is (its observed site, as the plan saw it, so a part
that slid in the grip is seen where it is) with where it was to be set down (the bin floor, the
top of what already lies in the bin under it, or where it was picked from): more than 20 mm above
it, the part is **not let go** (a tool or a part caught on a bin's rim also takes the load off the
wrist: a 110 x 80 mm box into a 130 x 100 mm bin stopped 96 mm above the floor and used to fall
98 mm at 1.3 m/s), the step fails
(`criterion_not_met`, with `held_high_mm`) and its outcome says why in a plain caveat. The old
lowering stopped when the tool pushed the part down by 5 N and opened 12 mm above the floor; a
cup's lip pushes its part away from the cup while the vacuum holds it (23.5 N on the 600 g item),
so that guard never lowered the suction gate's item and let it go 145 mm above the tote floor (it
hit the floor at 1.6 m/s, 2.2 to 3.3 mm deep for 10 to 16 ms: `physics_unreliable` under the
all-pairs rule).

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
pre-shape (so no finger shoves the panel), back off 40 mm from the face, rise 60 mm. An at_most
alone on a lid's angle takes it all the way shut (S2f): let go above its stop, a lid falls onto
it.

**A lid's lip** (S2b) has nothing to grip from above. A two-finger head whose opening fits the lip
tilts and pinches it from the front (studio v2, section 12); a pull on to a `force_n` at_least goal
then goes on along the lid's arc with the pitch held at the goal's, the arc the search's
swept-volume check covers (S2f: it used to turn on with the lid, to -44 deg on a 40 mm lid at 8 N
where the check ends at -13 deg). The pinch is taken only when the palm stays 12 mm inside its
travel along the whole pull (past vertical it sits the tool's length beyond the free edge): a 240
mm lid at y = 150 mm, whose pinch commanded the palm to 365 mm against the slide's 350 mm, is
hooked instead, and the plan's note says why. Any other tool hooks it: the tool's
calibrated sideways hook is turned
toward the hinge, lowered outside the free edge (10 mm out, plus whatever of the tool reaches
further toward the lid than the hook), moved in under it, guarded, by the hook's depth less 3 mm
(at most the lip less 2 mm and 40 mm), raised 3 mm onto the underside and carried along the arc,
open. The soft arms' horizontal arms are such hooks (131, 151 and 171 mm); grippers, hands, cups
and the designed sliding hook gripper (whose hooks point inward, 4 mm) have none, reach nothing
under the lip and fail honestly.

**A door's bar** (studio v2) is gripped from the front only: a two-finger head whose opening fits
the bar, the tool tilted 90 deg and normal to the face, its closing axis across the bar ("The door
bar front pinch", section 12). Nothing grips it from above. A tool that does not fit the bar, or
whose front grip the palm could not follow, is not attempted and the step fails plainly (section 12).
The Task Engineer's layout fixer moves a door along x into the front grip's reach
([STUDIO_ENGINEER.md](STUDIO_ENGINEER.md), the door-bar rule).

**Reach** (section 14): a grip checks its engagement, its whole pull and a pull on to a `force_n`
goal against the palm's travel (+-350 mm less 5 mm), and a hook the same from under the lip, less
the pinch's 12 mm; a step that would leave it is not attempted and fails `out_of_reach` with a
plain reason. A door bar's front grip checks its draw back as well, less the same 12 mm. The
retreat after a grip or a hook (the grip's 40 mm off the face, the hook's draw back out from under
the lip) never refuses a step: it stops at the travel less the same margin, then the tool rises. A
drawer at the table's edge is opened this way: its full retreat would put the palm at 383 mm. A
tool without a hook lifts nothing, so its pull on to a force goal runs along the lid's whole arc and
past its stop: that is how the hook of the 240 mm lid at y = 150 mm with a 20 N goal commanded the
palm to 368.7 mm.

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
outer face 15 mm in from the free edge along the arc to the goal (3 deg past it); where a bar
handle stands off the face (S2e), a lid is pushed on the bar's outer side, 5 mm on the hinge side
of it (the tool reaches from its tip toward the free edge, so it meets the whole bar), and a door
at the middle of its outer face, clear of the bar that runs up it. A lid open past vertical that
must close (a goal below 80 deg, no `force_n` at_least) is **carried** when the tool can hold it
(S2f): a two-finger gripper whose fingers fit the lid's thickness pinches it from the front as
`hook_pull` pinches a lip and carries it down its arc at the step's peak speed (at least 60 mm/s
[assumed]), guarded (it stops advancing on anything but the lid over 5 N, or on the lid over the
push limit; held back for 0.3 s it lets go there), lets go, draws back off it, rises until the
tool hanging straight down clears everything, and turns back. What it holds is sized from what
lies under the lid (every other object and the table, seen as the transit height is seen; the
pre-shaped fingers swept along the carry with 2 mm kept clear [assumed]), in order:

* the free edge, its tips in by the lip's depth less 4 mm (or stopping 4 mm short of a bar
  handle), less where needed so the lower finger clears what the edge overhangs: such a lid is
  carried to its stop (the H1 lid, whose 20 mm lip overhangs its tote);
* a bar handle, pinched across with the inner finger between the bar and the face, when the
  fingers fit there (the fingers then ride above the lid, so a lid flush with its box closes all
  the way);
* else the free edge, carried down to 2 deg above where its lower finger would meet the box, and
  let go there: a lid flush with its box (a default handle lid on a default bin: let go at 7 to
  12 deg, it falls the rest from 1.1 to 5.8 deg at 124 to 259 deg/s) falls shut, and its step says
  so (below). A bar handle with room under it (50 mm) is held by the bar instead: carried to its
  stop.

The whole carry must keep the palm within its travel (at the lid's stop the palm sits the tool's
length beyond the free edge, so a deep lid near the front of the table takes it past 350 mm) and
the plan must fit a step (30 s); the palm is kept 12 mm inside its travel on the whole path (the
wrist's 10 mm yield along the tool and 2 mm of aim error, `hook_pull.PINCH_REACH_MARGIN`), and the
search's tilted poses check the same hold and the same path (`push.lid_hold`,
`hook_pull.pinch_reach`). Otherwise, or with any other tool, the plan's note says
why and the lid is pushed only to 80 deg; its weight closes it (a push from outside cannot hold
it back: the frozen H1 lid met its stop at 340 to 400 deg/s), and the tool then presses the closed
lid home from above, at the middle of its outer face.
**Approach** (S2e): a transit to 30 mm
above the standoff point, then a guarded move in over 0.8 s (it stops on contact with any object
over 5 N or the push limit, and the push then starts where it stopped); the standoff, 10 mm, is
the whole tool's (whatever reaches further toward the face than the contact point, a wrist flange
or a long finger, stands off it too), and the push meets the face, and its object its stop (the
last 20 mm of the travel), at 15 mm/s at most [assumed], so an impact stays under the limit. A
drawer is pushed on the front of its handle along the
slide (3 mm past the goal); a part is pressed down on its top until the press force (10 N, or 60 %
of the limit; a `force_n` at_least above that, S2d) for 0.5 s; a plug in its socket pressed down
to the goal depth. The push advances at `speed_mm_s` while the tool's net force is under the limit
(80 % of `max_force_n`, else 30 N [assumed]) and holds while it is over. A push to a criterion on
the object's state (S2d) ends when the **observed** state meets it (1 deg, 1 mm or 0.2 mm inside
its bounds [assumed]), not when the command arrives: the wrist servo lags behind the command under
the object's resistance (a drawer pushed shut stopped 7.7 mm open against a 5 mm criterion), so
the push goes on, up to the limit over the servo's stiffness plus 2 mm past its path (`over`),
and while the state is still unmet on past that, by 60 mm at most [assumed] (section 14: the
tool's own give takes travel too; the gripper's fist still flexes under contact). A
`force_n` at_least is a goal of the last push on top of its travel: it presses on, within that
reach and while the observed state still meets its criteria, until the force is reached and held
0.5 s, within the equal budget `force_press` plus that travel. A push whose observed state meets
its criteria while it is held at its limit (shut against its stop) is done. Then the wrist target
comes back to where the wrist is (the push stops), hold 0.3 s, back off 30 mm, rise 60 mm.
With `approach = front` (studio v2, section 12) a push on an upright face tilts the tool at it.
A push whose tool the palm cannot put on the face where the push starts or ends fails
`out_of_reach`, and a button the step's limit cannot latch is not pressed (section 14).

Observed: the template state at the end, `force_n` (peak net force), `penetration_mm`; on a lid,
sampled every physics step (a lid falling at 385 deg/s crosses its last 2 deg in about 5 ms, between
two check ticks), `lid_release_deg` (the angle a lid that fell shut fell from: where it last closed
faster than 5 deg/s with nothing of the tool on it; 0 when it never fell) and `lid_stop_dps` (the
fastest it closed within 5 deg of its stop). A lid that met its stop faster than 30 deg/s
[assumed] is not a failure (a lid with a catch snaps shut), but its step's outcome carries a plain
caveat ("The lid fell shut from 5.8 deg and met its stop at about 260 deg/s: the tool let go of it
above its stop").


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
| push approach (S2e) | 0.8 + touch | the guarded move into a push's standoff (0.8 s), plus 1.3 x the 35 mm a push covers at the touch speed (the standoff, 5 mm into the face, the last 20 mm) at 15 mm/s instead of `speed_mm_s`; a tool whose flange or body overhangs its contact point adds 1.3 x that overhang at `speed_mm_s` (it starts further out) |
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
| grasp | steel slug into a tote | two-finger adaptive, pinch | 14/14 | rigid-pad gripper A (4.1 N grip) | 0/14 (criterion not met) | 0.43 mm | 12.6 s |
| suction | 600 g flat item into a tote | 25 mm cup | 14/14 | 10 mm cup (4.7 N hold) | 0/14 (criterion not met) | 0.11 mm | 12.8 s |
| wrap | lift a flanged bushing | 16-segment soft arm | 14/14 | 16 mm cup (bore) | 0/14 (criterion not met) | 0.27 mm | 10.6 s |
| hook_pull | drawer by its bar, 120 mm | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (slip) | 0.22 mm | 11.0 s |
| twist | handwheel a quarter turn (3 N m) | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (slip) | 0.20 mm | 9.5 s |
| insert | 20 mm plug, 0.4 mm clearance, 2 mm chamfer | two-finger adaptive, power | 13/14 | 12-segment soft arm | 0/14 (jam) | 0.57 mm | 18.2 s |
| push | press a box down (5 N) | two-finger adaptive, pinch | 14/14 | 12-segment soft arm | 0/14 (criterion not met) | 0.06 mm | 6.0 s |
| insert (S2b) | 8 mm plug, 30 mm, 0.2 mm, 1 mm chamfer, to 15 mm at 20 N | fin-ray pad gripper | 14/14 | two-finger adaptive, power | 0/14 (physics unreliable) | 0.81 mm | 18.5 s |
| insert (S2b) | the same, rigid-pad gripper A | rigid-pad gripper A | 11/14 (jam: its 4.1 N grip lets the plug slide) | | | 0.62 mm | 18.1 s |
| insert (S2b) | the same port, 12 mm plug | rigid-pad gripper A | 13/14 | two-finger adaptive, power | 0/14 (jam) | 0.38 mm | 18.2 s |
| hook_pull (S2b) | free 380 x 280 lid by its lip to 90-110 deg at 20 N | 12-segment soft arm | 14/14 | | | 0.44 mm | 24.6 s |
| hook_pull (studio v2) | the same lid, pinched with the wrist tilted (section 3.2) | two-finger adaptive, pinch | 12/14 (2 constraint exceeded) | | | 0.58 mm | 25.5 s |
| hook_pull (S2b) | 400 x 300 door by its handle to 80 deg at 30 N, off the home pose | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (constraint exceeded) | 0.26 mm | 18.2 s |
| hook_pull (studio v2) | the same door, its bar gripped from the front with the wrist tilted (section 12) | two-finger adaptive, pinch | 14/14 | 25 mm cup | 0/14 (criterion not met: not attempted, not a two-finger gripper) | 0.21 mm | 17.4 s |

The insert step times grew by 1.8 s (16.4 s in S2): every tool on a port now carries the same
press-home budget (section 4).

Rerun after S2f (2026-10-26), under the all-pairs penetration rule (section 3): the three pick rows
with the set-down (section 3.1). Their step times grew by 1.2, 1.2 and 0.2 s (the guarded descent
onto the support); the suction reference was 1/14 under the all-pairs rule before it (its 600 g
item let go 145 mm above the tote floor, `physics_unreliable`), and is 14/14 again with no flagged
transient. The lip row of the two-finger adaptive gripper is the pinch (studio v2): it was the
wrong effector, 0/14 (slip), when it could only hook. The door row's reference grips the bar from
the front (studio v2, section 12, 2026-10-29): 81.7 to 87.2 deg, 1.3 to 8.0 N; from above it had
reached 87.3 to 88.6 deg. Nothing grips a door's bar from above now, so the cup, which is not a
two-finger gripper, is not attempted (`criterion_not_met` in 0.2 s, with the reason).

No MuJoCo reset (`physics_invalid`) in any gate run. Two in-process runs and a spawned worker
give the same `traj_sha` for every gate task.

**Every template variant** (seed 0, the two-finger adaptive gripper; `power` for the plug
inserts):

| Variant | Step | Result | Measured |
|---|---|---|---|
| lid, lip | hook pull to 80 deg | fails (slip) | a straight finger cannot get under a horizontal lip; the 12-segment soft arm hooks under it: 110 deg, slip 6.2 mm (S2b) |
| lid, handle | hook pull to 80 deg | passes | 110 deg (on its stop), slip 11.7 mm |
| door, lip | hook pull to 60 deg | passes | 68 deg, slip 0.4 mm (edge pinch) |
| door, handle | hook pull to 60 deg | passes | 66 deg, slip 1.1 mm: the bar gripped from the front (section 12; from above, 68 deg and 0.1 mm) |
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
  horizontal arms) and room under the edge, or pinched by a two-finger gripper with the wrist
  tilted (studio v2): a lid flush with, or inset in, its bin's walls cannot be opened by any
  top-down tool (section 8).
* **A lid pushed shut falls unless the tool can hold it all the way** (S2f). A two-finger gripper
  carries a lid open past vertical down to its stop (section 3.5) where its whole carry stays in
  the palm's reach and something of the lid clears what lies under it. A cup, a hand, a soft arm,
  or a gripper whose palm would leave its travel (at the lid's stop it sits the tool's length
  beyond the free edge; at the start it sits as far back past the hinge, so lids near the back or
  the front of the table, and deep lids, are out of reach) pushes it over and it falls onto its
  stop: 340 to 400 deg/s on the frozen H1 lid before S2f, 1.8 mm into its tote's rim for 8 ms on
  the lidded-tote test lid (a flagged transient). A lid flush with its box, held by its edge, is
  let go a few degrees up (7 to 12 deg) and falls the rest at 124 to 259 deg/s (`lid_stop_dps`).
  Each such step carries the fall caveat. Carried to its stop, the H1 lid meets it at about
  5 deg/s.
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
when its press from above starts (fixed in S2e, section 11). The engineer's
check that a `torque_nm` at_least stops a twist on a latch or valve before it turns
(`rtt.studio.engineer.checks`, not an S2 file) still holds for a twist with only a torque goal; a
twist with a rotation goal as well now turns to it.

## 11. S2e: a push's approach

**The bug.** Closing a lid with a bar handle from past vertical (opened to 110 deg, then
`angle_deg` <= 3 under a 40 N limit) ended `constraint_exceeded` for most grippers, and the
tool was already pressing 43.7 N on the lid before its press started. The push planned every
standoff off the panel's **face**, at 15 mm in from the free edge, which is where the bar handle
stands (centred 20 mm in, 20 mm of clearance under it plus its diameter off the face); and the
moves into the standoff were position-controlled. The press from above landed on the bar and the
z servo drove 11 mm into it (44 N); the arc push's move in ran into the bar at transit speed
(57 N, 58 N, 104 N for the servo, rigid-pad and linkage grippers). Three more causes of the same
kind came up on the way: on a door the move down beside the face landed on the bar's top end
(150 N, the z servo's limit); a long finger or the wrist flange reached the face before the
contact point, which the standoff did not count; and a push met the face at `speed_mm_s` (60 mm/s
in H1, 24.8 N of impact against a 20 N limit), then held its command past the stop through the
hold and the retract (36 N).

**The fix** (`primitives/push.py`, section 3.5): where a bar stands off the face, a lid is pushed
on the bar's outer side, on its hinge side, and a door at the middle of its face; the press from
above is at the middle of the closed lid; the standoff is the whole tool's, flange included; the
move in is a guarded move (`Guarded` gained a per-move force, the push uses the guard force or the
push limit, whichever is lower), and a push stopped short starts from where it stopped; the push
meets the face and its stop at 15 mm/s; a push whose criteria are met while it is held at its
limit is done; and the push stops (the target comes back to the wrist) before the hold and retract.

Pushed shut from 110 deg (`angle_deg` <= 3, 40 N), opened by the adaptive gripper, three seeds
per tool (`test_a_lid_opened_past_vertical_is_pushed_shut_within_the_step_force_limit` runs the
rigid-pad, servo and linkage grippers):

| Tool | Lid with a bar, before | After | Door (400 x 300 mm, bar) from 88 deg, before | After |
|---|---|---|---|---|
| two-finger adaptive | 3/3, 10 N | 3/3, 10 N | 0/3, 150 N | 3/3, 32 N |
| three-finger adaptive | 3/3, 10 N | 3/3, 10 N | 2/3, 75 N | 3/3, 32 N |
| fin-ray gripper | 0/3, criterion not met | 0/3, criterion not met (its fingers bend; 27 N peak) | 0/3, 336 N | 3/3, 32 N |
| rigid-pad gripper A | 0/3, 58 N | 3/3, 10 N | 0/3, 167 N | 3/3, 32 N |
| 25 mm cup | 3/3, 29 N | 3/3, 10 N | 0/3, 150 N | 3/3, 33 N |
| 16 mm cup | 3/3, 28 N | 3/3, 10 N | 0/3, 150 N | 3/3, 32 N |
| 12-segment soft arm | 3/3, 10 N | 3/3, 10 N | 3/3, 8 N | 3/3, 1 N |
| library servo gripper, 110 mm | 0/3, 57 N | 3/3, 10 N | 0/3, 351 N | 3/3, 32 N |
| library linkage gripper | 0/3, 104 N | 3/3, 13 N | 0/3, physics unreliable | 3/3, 32 N |

Peaks are the predicate's `force_n`. A lid with a lip (no bar) ends as before: 24/27, the fin-ray
gripper 0/3 both times, every peak at most 13 N. A door is pushed 3 deg past closed, so a push
ends held at its 32 N limit against the frame; a press (10 N) ends a lid's close.

**H1's `close_lid`** (frozen plan: push, `angle_deg` <= 5, 20 N limit, after the lid is opened
to 110 deg by its lip), three conditions: the recommended head's 16 mm cup closes it 3/3 before
and after, its peak 14.8 to 15.0 N before and 10.3 N after (the press force; before, the impact
at 60 mm/s). The best single (the library servo gripper) never reaches `close_lid`: it fails
`open_lid` (slip, or 21.3 N against the 20 N limit), the same before and after. From the
recommended head's state before `close_lid` (lid open, parts placed), the servo gripper closes
the lid 1/3 before (2 x `constraint_exceeded`, 24.8 N: its impact at 60 mm/s) and 3/3 after
(10.1 N).

**Gate.** The S2 gate's push and hook_pull rows end as before (14/14 with their references,
0/14 with the wrong effectors, the same failure modes and penetrations); the push step's median
time grows from 5.3 to 6.0 s (the touch speed). The S2d probes end as in section 10.

## 12. Studio v2: the wrist pitch in the primitives

The contract's wrist gained a pitch (STUDIO.md 4.1: +-90 deg about the palm's x axis between the
yaw and the roll, 180 deg/s). The primitives use it as follows (`primitives/_common.py`,
`hook_pull.py`, `push.py`).

**Six targets.** Every controller returns all six wrist targets (`_common.wrist_target(point,
tool_point, yaw, roll, pitch=0)` puts a wrist-frame tool point at a world point through
`scene.wrist_rotation`, `_common.start_pose(ctx)` is the step's six-value start). Every motion
keeps the tool straight down except the two below. A move that changes the pitch lasts at least
1.1 x the min-jerk time the scene's own pitch cap allows (`Path.start(pose, pitch_dps)`,
`Path.to`), so a step that starts tilted turns back to straight down within the cap in its first
move. Proprioception reads the pitch joint too
(`tool_point_world` is `scene.world_tcp`, `_common.wrist_frame`). `StepContext.wrist_start` stays
five values beside `wrist_pitch_start` (the five-target runner rule is unchanged).

**The lip pinch** (`hook_pull`, mode `pinch`). A lid's lip (`orientation = lid`, `feature = lip`)
is pinched by a two-finger gripper (`two_finger`: a parallel or part-shaped jaw, a catalog gripper
with two fingers, parallel, angular or adaptive, a designed fingered hand with two) whose
calibrated open aperture clears the lip by 6 mm a side and whose shut aperture is under the lip's
thickness (`lip_pinch_fits`); any other tool hooks it as before, hands with three or more
fingers, hook grippers and tentacles included (their apertures would clear a 6 mm lip, but they
are not pinching tools). The yaw sets the pitch axis along the hinge (`pinch_yaw`), the
pitch is 90 deg so the tool points along the lid toward the hinge with its closing axis across
the lid's thickness, the roll goes to the nearest half turn. The wrist rises to the transit
height, turns (within the pitch and roll caps, 1.1 x margin), descends with the tips 20 mm out
from the free edge at the lip's height, pre-shapes as closed as still clears the lip, and moves in
from the front, guarded (5 N on any object), until the tips are the lip less 4 mm under the
overhang (at most 25 mm: 16 mm on the hero tote's 20 mm lip) [assumed]. It closes (1.0 s plus
0.3 s), then carries the pinch point along the arc to the pull's goal with the pitch at 90 deg
less the lid's angle (`pinch_pitches`), so the grip turns with the lid; the carry, and a force
goal's pull-on after it, are no faster than the pitch cap allows (a 60 mm lid at 200 mm/s slows
to it), and a start a hair past the range (a lid observed just below closed), within 2 deg, is
clamped to it. While pinched the wrist
yields along the tool axis to the force the tool puts on the lid along it, 5 mm per N s, at most
10 mm [assumed; force feedback through the wrist command]. At the goal (a lid that must stay open
goes to 103 deg for an at_least of 95) it opens to the pre-shape, draws back along the tool axis
off the lip (20 mm past the free edge), rises 60 mm while the pitch turns back to straight down,
timed from the pitch where it let go.
A goal the pitch cannot follow within +-90 deg, or a tilted approach that would put the palm past
its travel (+-0.35 m: the palm sits the tool's length out from the lip), falls back to the hook.
Slip is measured as for a hook, the pinch point in the panel's frame.

**The door bar front pinch** (`hook_pull`, mode `bar`). A door's bar handle (`orientation = door`,
`feature = handle`: a vertical bar standing off the outer face near the free edge) is gripped from
the front only. The gripper must be of the lip pinch's class (`two_finger`), with a calibrated open
aperture that clears the 10 mm bar by 6 mm a side and a shut aperture under it (`bar_fits`). The
pitch is 90 deg, so the tool is horizontal. The yaw points it at the door along the face normal
(`bar_yaw`, the lip pinch's yaw with the hinge vertical). The roll goes to the nearest quarter turn
(`bar_roll`), which lays the closing axis horizontal along the face, across the bar. The wrist
rises to the transit height and turns there, within the pitch and roll caps. It descends with the
tips 25 mm out from the bar's front at the bar's mid-height and pre-shapes as closed as still clears
the bar. Then it moves in along the tool axis, guarded (5 N on any object), until the tips are past
the bar's front by the bar's depth off the face less 4 mm (26 mm on a 10 mm bar 20 mm off the face),
so they stop 4 mm short of the face [assumed]. The depth counts how far the tips reach beyond the
TCP as the fingers close (`bar_depth`, `bar_drop`). It closes (1.0 s plus 0.3 s) and carries the bar
along the door's arc to the pull's goal. The heading turns with the door, so the tool stays normal
to the face, no faster than the 180 deg/s turn cap allows, and the wrist yields along the tool axis
as in the lip pinch. At the goal it opens to the pre-shape, draws back along the tool axis until the
tips are 30 mm clear of the bar, and rises 60 mm while the pitch turns back to straight down
[assumed].

The palm sits the tool's length out from the bar along the face normal, and a door that opens toward
a table edge carries it there. So the grip, the whole pull (a pull on to a force goal included) and
the draw back must keep the palm 12 mm inside its travel (`bar_reach`). Nothing else engages a
door's bar, and nothing grips it from above. A tool that does not fit the bar fails the step
`criterion_not_met` in 0.2 s ("Cannot grip the door's bar from the front: its calibration finds no
pair of fingers either side of its axis to close across the bar; the step was not attempted"). A
front grip the palm could not follow fails it `out_of_reach` ("Out of reach: the front grip's pull
would take the palm to x -454 mm, y 9 mm, past its +-350 mm travel less 12 mm; the step was not
attempted"). Neither step is attempted. The search's tilted poses make the same reach check
(`search.tilted_poses`), so a two-tool head's swept-volume check sees the tilt. The Task Engineer's
layout fixer moves a door that neither reference gripper on a tool changer could grip from the front
along x until one can ([STUDIO_ENGINEER.md](STUDIO_ENGINEER.md), the door-bar rule).

**Measured** (nominal or seed 0 unless a grid is named; `test_a_door_bar_is_gripped_from_the_front_on_the_14_gate_conditions`,
`test_a_door_bar_the_front_grip_cannot_reach_or_fit_is_not_attempted`):

| Door | Tool | Result |
|---|---|---|
| S2b door, 400 x 300 x 10 mm at (0, 150 mm), 80 deg within 30 N, the 14 gate conditions | two-finger adaptive, pinch | from the front, 14/14: 81.7 to 87.2 deg, 1.3 to 8.0 N, penetration at most 0.21 mm, slip 1.5 to 5.5 mm, 17.4 s; at the grip the closing axis is square to the bar (0.0 deg) and the tool axis within 0.13 deg of the face normal (2.9 deg at worst during the pull), pitch 90 deg |
| the same door | servo-110 | not attempted, `out_of_reach`: the front grip's draw back would take the palm to x -365 mm (from above it had opened it to 88.0 deg) |
| the same door | angular-jaw (hobby arm), 25 mm cup | not attempted, `criterion_not_met`: no pair of fingers to close across the bar; not a two-finger gripper |
| default door, handle, at (0, 80 mm), 60 deg | two-finger adaptive, pinch | from the front: 65.9 deg, slip 1.1 mm (from above before: 67.9 deg, 0.1 mm) |
| hotel door, 400 x 300 x 20 mm at (-140, 290 mm), where its recorded plan first stood | two-finger adaptive, servo-110 | not attempted, `out_of_reach`: the front grip's pull would take the palm to x -407 and -454 mm (from above they had opened it to 87.9 and 88.1 deg) |
| the same door | angular-jaw (hobby arm) | not attempted, `criterion_not_met` (from above it had opened it to 89.1 deg) |

At x -140 mm the hotel door is hinged on its left and swings its face toward the table's left edge.
Gripped from the front, the palms of the eligible grippers (113 to 181 mm from their tips) would
leave the travel at 63 to 74.5 deg of the 88 deg goal. The layout fixer now moves it to x 90 mm,
where the 110 mm servo gripper on a tool changer reaches it, and moves the tray out of its swing.
There the hotel job is done from the front (`STUDIO_ENGINEER.md`, the hotel smoke run).

Letting go, the adaptive gripper's distal links spring straight as the fingers open. They had
curled round the bar onto the face while holding it, and they strike the face. The door, pulled to
87.7 deg, swings back to 81.7 deg on seed 4; the 8 deg goal margin covers it. Drawing back while the
fingers open instead pulled the doors on to 97 to 99.6 deg, which would bring a door with a 90 deg
stop onto it fast. So the order stays: open, then draw back.

**Unchanged runs.** `traj_sha` before (8620c7a) and after, on the S2 gate (7 primitives, reference
and wrong tool, 14 conditions each), the lip rows, H1 (8 conditions, 4 steps), the lidded tote, the
set-down picks, the sliding-door, machine and far-reach template rows, the hotel door (x -140 and
-170 mm with the adaptive, servo-110 and angular-jaw grippers), the S2b door (the adaptive gripper
on 14 conditions, three other tools on seed 0) and the default door with a lip and with a handle:
254 of 276 steps are identical. The 22 that changed are all door-bar runs. 15 are now gripped from
the front: the S2b door's 14 and the default door's handle. 7 are now not attempted, where the bar
was gripped from above before: the S2b door with the servo-110, the angular jaw and the cup, and the
hotel door at x -140 mm with all three grippers. The seventh is the angular jaw at x -170 mm, whose
step was already not attempted, out of reach; it now fails as a misfit, with the same trajectory.

**The push from the front** (`push`, `approach = front`). A push on a face whose normal is within
30 deg of horizontal tilts the tool 90 deg, the yaw square to the face (it follows a door as it
turns), and pushes with the tool's tip on its axis (the lowest point of its calibrated footprint
at the push closure); nothing of the tool reaches further toward the face, so the standoff is
10 mm. The transit turns the pitch at the transit height, the guarded move goes in horizontally,
the push runs as before, and the retract backs off 30 mm, rises until the tool hanging straight
down would clear every object by 30 mm (at least 60 mm), and only then turns the pitch back (a
tilted tool's tip swings down by its length as it turns). A face the tilted palm cannot reach is
pushed from the top. Faces pushed from above ignore it. A required `approach` row would have
invalidated every frozen plan's push step, so `approach` is an optional push param (section 13.5):
omitted at its default `top`, read as `step.params.get("approach", "top")`.

**Measured** (frozen H1 plan, `h1_tote`, its 8-condition grid, `sub_task(open_lid)` and the full
task, a single tool):

| Tool | `open_lid` | Peak force | Full H1 |
|---|---|---|---|
| library servo gripper, 110 mm | 8/8, 110 deg (its stop) | 2.93 to 2.94 N | 8/8, 66.4 s cycle |
| library adaptive gripper, 85 mm | 8/8, 110 deg (its stop) | 6.7 to 9.1 N | 0/8 (`pick_pack` not in the tote, not a pitch matter) |

The adaptive 85 mm gripper passes `open_lid` only with the yield along the tool axis (without
it: 6/8, 2 x `constraint_exceeded` at 20.3 to 21.1 N, prototype run). The servo gripper's
trajectories equal the prototype's run for run.

**Gate.** The S2 gate table re-run on its 14 conditions (seeds 0 to 9 and placement errors of
2 mm in +x, -x, +y and -y, `test_the_s2_gate_table_holds_on_its_14_conditions`), before (the
Studio v2 contract) and after:

| Primitive | Reference, before | After | Wrong, before | After | Max penetration on success, after |
|---|---|---|---|---|---|
| grasp | 14/14 | 14/14 | 0/14 (criterion not met) | 0/14 (criterion not met) | 0.36 mm |
| suction | 14/14 | 14/14 | 0/14 (criterion not met) | 0/14 (criterion not met) | 0.08 mm |
| wrap | 14/14 | 14/14 | 0/14 (criterion not met) | 0/14 (criterion not met) | 0.27 mm |
| hook_pull | 14/14 | 14/14 | 0/14 (slip) | 0/14 (slip) | 0.23 mm |
| twist | 14/14 | 14/14 | 0/14 (slip) | 0/14 (slip) | 0.23 mm |
| insert | 13/14 | 13/14 | 0/14 (jam) | 0/14 (jam) | 0.56 mm |
| push | 14/14 | 14/14 | 0/14 (criterion not met) | 0/14 (criterion not met) | 0.06 mm |

Every trajectory but insert's is the same bit for bit (none of them tilts); insert's 28 differ
in the last digits because its alignment now reads the tool's frame with the pitch joint's
servo deflection. The insert failure is the same condition (+2 mm in y, `jam`). The S2b row
"free lid by its lip, two-finger adaptive 0/14 (slip)" no longer holds by design: the adaptive
gripper's opening fits a 5 mm lip, so it pinches it
(`test_a_lip_without_a_hook_or_without_room_under_it_stays_shut` keeps the soft arm on a lid inset
in its rim; `test_a_two_finger_gripper_tilts_and_pinches_a_lid_lip` replaces the adaptive case).

## 13. Studio v2: four job templates

Four templates for the real-job heroes (solar install, tile setting, machine tending, a call
button): `src/rtt/studio/templates/{panel_rail,tile_bed,spring_button,sliding_door}.py`, tests in
`tests/test_studio_templates.py` (each builds, rests without drift, reports its quantities, its
params and physics, and the S2 rule: a fitting tool succeeds where a wrong one fails). They run in
the Studio now and are **not offered to the Task Engineer yet** (`ObjectTemplate.planner = False`,
and their quantities and the push's `approach` likewise): its prompt, tool schema and web catalog
render `types.planner_templates()` and its siblings, so the recorded prompts, the tool hash and the
committed web catalog are unchanged until the Engineer is taught them. Every default in the tables
below is [assumed] except the panel's 1000 x 600 x 35 mm and 8 kg and the tile's 300 x 300 x 10 mm
(the heroes' sizes) and the standard sizes cited (ISO 4032, ISO 261).

### 13.1 `panel_rail` (approximation)

| Item | Model |
|---|---|
| Panel | a free box `length_mm x width_mm x thickness_mm` (default 1000 x 600 x 35 mm) of `mass_g` (8000 g) resting on the rails; the template's last body (the one a pick lifts) |
| Rails | two static rails along local y at x = +-length/4, 40 x 40 mm [assumed], each with an end stop 10 mm along y and 20 mm above the rail [assumed] at the panel's +y edge, `start_gap_mm` (2 mm) from it at the start |
| Clamp | at the middle of the panel's -y edge, 25 mm beyond it [assumed], on a short bracket rail (ending 5 mm short of the panel): a 20 mm post rising 5 mm above the panel's top [assumed] (a socket over the clamp stays above the panel) carrying the `threaded_joint` model (`threaded_joint.add_thread`, the same code): a 13 mm across-flats hex (the M8 nut's size, ISO 4032), **modelled as a tall 13 mm hex so the simulated socket can seat; a standard M8 nut is 6.8 mm tall** (15 mm [assumed]; the nut runner engages members at least 12 mm tall), on an M8 coarse thread (1.25 mm, ISO 261), three turns from the thread start to seated [assumed], a 1 mm lead-in chamfer, its friction ramp from 0.3 N m [assumed] to `clamp_torque_nm` |
| `clamp_torque_nm` | default 6.0 N m [assumed], at most 10 (the nut runner's clutch cap), below the nut runner's 6.7 N m rating. The clamp does not press the panel: seating it is the clamping |
| Reach | the pose is midway between the panel's top centre (a frame's lift) and the clamp, on local y: each is `(width/2 + 25 mm)/2` (162.5 mm) from it, so both stay inside the palm's +-350 mm travel for every pose within +-182 mm on x and y at any yaw (tested at the poses and corners of 13.7). A twist or an insert whose target the palm cannot reach fails `out_of_reach` |
| Clamp latch | a light, gravity-compensated slide with no spring (0.02 N friction, 20 N s/m damping [assumed]) that a tendon pushes along as the clamp goes down its thread; it stays, so it records the deepest the clamp went. A tightened clamp springs back 0.03 to 0.06 mm off its seat stop when the socket lets go (measured); the latch still reads it tightened |
| Sites | `panel` (the panel's top centre), `member` (the clamp's top), `seat`, `flat` (aux) |
| State | `gap_mm` (the larger gap between the panel's +y edge and the two stops, at the rails), `clamp_torque_nm` (the ramp at the latch: 0 on the lead-in, 0.3 N m at the thread start, `clamp_torque_nm` on the seat stop; the clamp goes no deeper than a torque over the ramp takes it, so it is the torque it was tightened to), and the clamp's present `depth_mm`, `turns`, `angle_deg` (a threaded joint's, which twist and insert read) |
| Picks | the picks plan a part from `info["diameter_m"]` and `info["length_m"]`: a round stand-in 400 mm across, as thick as the panel (only its height places the tool; the frozen parts library caps a part at 0.5 m) |

Twist and insert reach the clamp through `templates.thread_info` (its `clamp_` keys under a
threaded joint's names) and `templates.member_body` (the clamp member, not the panel, is the v1
view's object body, which the nut runner's alignment reads).

### 13.2 `tile_bed` (approximation): a bed that keeps its set

The tamper's press is read after the pad lifts (the template state when the step ends), so an
elastic bed springs back and hides the press (Studio v2 measured the same lift for 5, 40 and
60 N on softened contacts). The bed therefore **keeps its set**: it is plastic. Reading the height
under load instead would need the push predicate to sample the state at a chosen moment of the
press (a change to a primitive another track is editing) and would make the result depend on how
hard the tool happens to press when the step ends.

| Item | Model |
|---|---|
| Tile | a free box `tile_mm` square (300), `thickness_mm` (10) thick, `mass_g` (2160 g), the template's last body; site `part` on its top centre (a push reads it there; no `height_m`) |
| Bed | a static substrate slab and a 3 x 3 grid of cells (90 % of their 100 mm pitch, 10 mm tall [assumed]) on vertical slides; their tops start `proud_mm` (2) above the tile's finished bottom, 20 mm above the table [assumed]; a cell sinks at most 3 mm below the level [assumed] |
| Plasticity | each cell's own actuator is a stiff spring (100 kN/m, damped 150 N s/m, armature 0.1 kg [assumed]) to an anchor that `passive()` moves each control tick only as far as keeps the spring within the cell's share of the yield (the threaded joint's stick-slip friction on a slide). The whole bed yields at `bed_yield_n` (30 N) plus `bed_hardening_n_mm` (15 N/mm) per mm of plastic sink [assumed], so a steady press P on a tile of weight W sinks it until yield + hardening x sink = P + W |
| State | `height_err_mm` (how far the tile's top centre stands from its finished level, above or below) and `tilt_deg` |
| Checks | an instance is refused (plainly) when the tile's weight is at or over the bed's yield (it would sink by itself) or its density is outside 800 to 4000 kg/m^3 [assumed] |

Measured (a steady press at the tile's centre for 1 s, then let go, `test_tile_bed_keeps_its_set...`):
unpressed it rests 1.85 mm proud (its 21.2 N is under the 30 N yield; the cells and contacts give
0.15 mm); 15 N leaves it 1.44 mm proud; 40 N seats it 0.23 mm below its level; 60 N over-presses it
1.56 mm below. Let go, it springs back 0.09 to 0.31 mm (the cells' elastic give and the contacts'
softness): the set is kept.

### 13.3 `spring_button` (rigid)

| Item | Model |
|---|---|
| Mount | `wall`: a plate 160 mm wide, 20 mm thick, reaching 60 mm above the button [assumed], its face through the pose facing local -y, the button's centre `height_mm` (120) above the table; `desk`: a 120 mm square console [assumed] `height_mm` tall with the button on top |
| Bezel | a square opening 1 mm wider than the button each side [assumed] |
| Button | a cylinder `diameter_mm` (40) across, 15 mm long [assumed], its face `proud_mm` (1) out of the bezel; a slide into the bezel, range `travel_mm` (3); a return spring preloaded to 30 % of `actuation_force_n` (4 N) against its rest stop, rising to `actuation_force_n` at its actuation point, 70 % of the travel [assumed] |
| Latch | a light slide with no spring (0.05 N friction [assumed]) that a tendon pushes along once the button passes its actuation point, and that stays: the call is recorded after the button springs back |
| Sites | `button` (its face centre), `face` (aux, the bezel at the button) |
| State | `travel_mm` (how far it is in now), `pressed` (1 once the latch moved 0.2 mm [assumed]) |

A button standing out less than its actuation point can be pressed only by a tool narrower than
the opening: a broad pad stops on the bezel. A push presses a button with at least 1.5 x its
actuation force (past its bottom-out at 1.3 x), within the step's limit (80 % of `max_force_n`).
A press whose cap is under the button's actuation force is not tried (section 14).

### 13.4 `sliding_door` (rigid)

A machine enclosure whose door slides sideways, for overhead loading: a front leaf over the doorway
and a roof leaf over the roof opening move together (the shape of a machine door that opens into
its roof), so a closed door blocks a load from above. The pose is the opening's centre (a fixture
`bin` at the same pose sits under it); the door slides along local -y to open (as a drawer pulls
out) and the machine's front faces local -x.

| Item | Model |
|---|---|
| Enclosure | static walls 10 mm and a roof 10 mm thick [assumed], `depth_mm` (300) by `height_mm` (250), long enough along y to park the open door over it; the opening is `width_mm` (250) along y and the interior's depth along x, through the roof and the front wall |
| Door | one body on a slide along local -y, range `[0, travel_mm]` (260), friction loss `pull_force_n` (10 N [assumed]), damping 2 N s/m (the drawer's); leaves 6 mm thick, 2 mm clear of the enclosure, 10 mm past the opening each side, 1200 kg/m^3 [assumed] |
| Handle | a fin 6 mm thick (along the slide) and 80 mm long standing `handle_mm` (40) up from the roof leaf at the opening's centre [assumed] |
| Sites | `door` (the travel reference), `handle` (the fin's centre), `front` and `opening` (aux, static) |
| State | `travel_mm` (0 closed), within the slide's stops |

It reuses the drawer's machinery (`types.SLIDE_KINDS`): the same slide info keys (`pull_x`/`pull_y`,
`travel_m`, `grip_m`, `push_out_m`, `front_gap_m`), and its fin is the drawer's `lip` handle (code
2: a plate standing up, gripped across its thickness along the slide, its face up).

### 13.5 The optional `approach` step param, and the frozen ids

`PrimitiveInfo.optional` holds params a step may leave out. A step omits one set to its default, so
a step written before the param existed hashes as it did. Push's `approach` (`top`, the default, or
`front`) is one: `Step.of(..., params={"approach": "front"})` carries it, `"top"` is dropped from the
step, anything else is refused. Hook_pull's `approach` stays the required row it has been since S1:
every recorded hook_pull step carries `"approach": "top"`, so dropping it would change their ids.
`test_frozen_hero_task_ids_are_unchanged` re-hashes the three heroes of use-case benchmark v1 from
the recorded plans and compares them with the ids pinned in `configs/usecase_v1.yaml` (h1_tote
e4586f36..., h2_coupling 33599e99..., h3_cabinet 3297a111...): equal.

### 13.6 Primitive changes

The new kinds reuse the existing controllers; the changes are kind sets and one branch:

* `_common.observed_state`: a sliding door reads like a drawer (its moving site is `door`); a
  panel's clamp reads like a threaded joint (`thread_info`).
* `hook_pull`: `_face_out` and `_closing_axis` take `SLIDE_KINDS` (a door's fin is a lip).
* `push`: a sliding door closes like a drawer; a **button** branch presses its face in along its
  axis, its travel and 2 mm more, until the press force holds: the push's `press` force (10 N, or
  60 % of the step's force limit, or its `force_n` goal) but at least 1.5 x the button's actuation
  force, within the step's limit. A tile is pressed by the part path.
* `twist`, `insert`: a panel's clamp turns and starts like a threaded joint (`axis_site`,
  `THREAD_KINDS`, `thread_info`, `member_body`). **Out of reach**: a twist whose engagement pose,
  or an insert whose planned moves or seat, would put the palm outside its travel (+-350 mm less
  5 mm) is not attempted: it holds still 0.2 s and fails `out_of_reach` (a new step failure mode),
  reporting `reach_x_mm`, `reach_y_mm` and `palm_limit_mm`, not a slip or a jam. Every other
  primitive checks its targets the same way (section 14).

### 13.7 The S2 rule on each

The cases of `tests/test_studio_templates.py`, the nominal condition and, for the fitting tools,
three more placement conditions of the 4-condition grid (`runner.condition_grid`):

| Template | Step | Fitting tool | Result | Wrong tool | Result |
|---|---|---|---|---|---|
| `panel_rail` | lift 60 mm, hold 1.5 s, set back (lift >= 25 mm, hold >= 1 s, gap <= 10 mm) | `vacuum-frame-2x3-600` | 4/4: lift 37.0 mm, hold 2.3 s, gap 2.0 mm, penetration at most 0.07 mm | a parallel gripper (84 mm) | fails: lift 0.008 mm (its fingers land on the flat panel) |
| `panel_rail` | the same lift | `vacuum-frame-2x3-600` | | `vacuum-frame-2x2-300` (61 N) | the mass decides: a 2 kg panel lifts 51.9 mm and holds 2.79 s; the 8 kg one lifts 0.01 mm |
| `panel_rail` | insert (depth >= 3.5 mm), then twist (clamp_torque >= 6.0 N m, 15 N m limit) | `nut-runner-socket-13` | 4/4: clamp 6.0 N m, wrist torque 5.96 N m, seated (7.69 to 7.75 mm after the release), penetration 0.53 mm; and at 8 more poses across the reach envelope (below) | the adaptive two-finger gripper | the twist slips: 0.20 N m, clamp 0.36 N m |
| `tile_bed` | press (force >= F, height_err <= 1 mm, tilt <= 0.5 deg, 60 N limit) | `tamper-pad-200`, F = 40 N | 4/4: 41.4 N, height_err 0.31 mm, tilt 0.00 deg | the same tamper, F = 10 N | fails: height_err 1.75 mm |
| `tile_bed` | lay: lift 40 mm, hold 0.8 s, set back | `vacuum-frame-2x2-300` | 4/4: lift 31.6 mm, hold 1.96 s | | |
| `spring_button` | wall, push from the front (pressed >= 1, 20 N limit) | the adaptive two-finger gripper | 4/4: pressed, 10.0 N | `tamper-pad-200` (its 200 mm pad) | fails: not pressed (the pad stops on the bezel) |
| `spring_button` | desk, push from the top | the adaptive two-finger gripper | 4/4: pressed, 10.2 N | | |
| `spring_button` | wall, a 15 N button at a 20 N limit, no force goal | the adaptive two-finger gripper | pressed at 16.3 N | | |
| `sliding_door` | hook_pull open (travel >= 200 mm), push shut (travel <= 5 mm, 40 N limit) | the adaptive two-finger gripper | 4/4: open 202.8 to 205.2 mm, shut 3.9 mm | `cup-25mm` | the pull slips: travel 0.0 mm |

The gripper's pull on the door reads `slip_mm` 31.6 to 33.3 mm: the adaptive fingers give 33 mm
along the pull as the force on the fin builds to the door's 10 N friction (a ramp, then the door
slides), so a step that bounds `slip_mm` needs a stiffer gripper or a lighter door.

**Reach** (`test_the_nut_runner_seats_the_clamp_across_the_reach_envelope`): the nut runner starts and
seats the clamp (6.0 N m, 7.75 mm) at panel poses (0, -60 mm), (130, 0 mm), yaw +30 and -30 deg,
and at the envelope's corners turned so the clamp lands at its extreme: (180, -180 mm, 0 deg),
(-180, 180, 180), (180, 180, 90) and (-180, -180, -90), the clamp 342.5 mm out on one axis; the
2 x 3 frame lifts the panel with its centre 342.5 mm out ((180, 180, 0) and (-180, 180, 90)). A panel
at (0, -300 mm) puts the clamp at y -462.7 mm: its insert and its twist each fail `out_of_reach` in
0.2 s (`test_a_clamp_out_of_reach_fails_as_out_of_reach_not_slip_or_jam`).

## 14. Studio v2: a push that falls behind, and reach for every primitive

From the machine-tending composite (section 13).

### 14.1 A push that falls behind its tool's give

The machine-tending task: a `sliding_door` at (0, 100 mm), a fixture `bin` 150 x 150 x 60 mm at the
same pose, a 40 mm part at (200, 100 mm), the adaptive two-finger gripper; open the door
(`hook_pull`, travel >= 200 mm), load the part into the fixture (`grasp`), push the door shut
(travel <= 5 mm, 40 N limit). At the door's default 10 N of friction the shut ended 40.1 mm open
(`criterion_not_met`) at every speed from 10 to 100 mm/s. It passed at 3 N, and at 10 N without
the load.

The cause is the tool's give. A push closes the fingers into a fist (closure 1); on this gripper
that drives both tendons to their 40 N limit with the tips pressed together, so the fist has no
lateral stiffness. After the pick lets go fully open, the fist settles differently and swings on
contact with the door (a proximal joint from 24 to 48 deg, the other finger to its -0.2 rad stop):
43 mm of the push go into the swing. The push is planned in position and went on only `over` past
its path while the door was still open (18 mm: the 32 N limit over the 2000 N/m servo, plus 2 mm),
then counted as arrived and let go.

A push to a criterion on the object's observed state now goes on past `over` while that state is
unmet, by `push.PUSH_ON_MAX` at most (60 mm [assumed]: the 43 mm measured, with 40 % to spare,
still short of the fist's 110 mm length). It goes on only while the force on the object is under
the step's limit, while the tool presses on nothing else harder than 5 N, and while the palm stays
within its travel, all within the segment's budget (unchanged). It goes at the step's speed while
the observed object is more than 20 mm from the segment's goal, then at the touch speed, so the
object still meets its stop at 15 mm/s at most. A push whose state is met within `over` runs
exactly as before (section 14.4). **The fist still flexes under contact**: only the travel it takes
is made up. A fist closure per family that does not saturate is not part of this change.

| Case (shut, nominal) | Speed (mm/s) | Before | After |
|---|---|---|---|
| load, 10 N | 20 (default), 40 | 40.1 mm open, fails | 3.9 mm, 10.4 to 11.6 N |
| load, 10 N | 60, 80, 100 | 40.1 mm open, fails | 6.3, 8.1, 8.8 mm, fails: the segment's budget runs out in the last 20 mm at the touch speed |
| load, 10 N | 10 | 39.9 mm open, fails | 29.3 mm, fails: a 10 mm/s push of 235 mm reaches the 30 s step cap |
| load, 3 N | 10 to 100 | 4.3 to 6.7 mm (fails at 10 and 100) | 3.9 to 4.5 mm, passes at every speed |
| no load, 3 N and 10 N | 10 to 100 | passes | the same `traj_sha` at all 12 points |
| S2d drawer shut, seeds 3 and 9 | 20 | 37.8 mm open, fails (both) | 3.9 mm, 6.7 N (both) |

The drawer of the S2d test fails the same way on seeds 3 and 9 (its fist swings on the drawer's
front).
`test_a_door_shut_after_loading_the_fixture_closes_within_5_mm` and
`test_a_push_that_loses_travel_to_its_fist_goes_on_until_the_drawer_is_shut` hold both.

### 14.2 Out of reach for every primitive

Only twist and insert used to check their targets against the palm's travel. The helpers now live
in `_common` (`reach_problem`, `hold_out_of_reach`, `out_of_reach_verdict`, `REACH_MARGIN`) and every
controller calls them at plan time. A step whose target lies outside the travel holds still 0.2 s
and fails `out_of_reach` with `reach_x_mm`, `reach_y_mm`, `palm_limit_mm` and a plain caveat ("Out of
reach: the pull would take the palm to x 0 mm, y 515 mm, past its +-350 mm travel less 5 mm; the
step was not attempted"), instead of running in vain to a silent `criterion_not_met` or a `slip`.

* **Picks**: the grasp pose, and the set-down over a target bin.
* **Hook pull, grip**: the engagement, the pull (every 2 deg of an arc) and a pull on to a `force_n`
  goal. **Hook** (a lid's lip): the same from under the lip, with the pinch's 12 mm margin
  (`hook_pull.PINCH_REACH_MARGIN`). The retreat after either (off the handle, or back out from
  under the lip) is not a reason to refuse: it stops at the travel less the margin
  (`hook_pull._retreat_within`), then the tool rises. A door bar's front grip (section 12) checks
  its draw back too, with the same 12 mm (`hook_pull.bar_reach`): where any of its moves would leave
  the travel the step is not attempted and fails `out_of_reach`, naming the front grip's move ("Out of
  reach: the front grip's pull would take the palm to ...").
* **Push**: the tool on the face where each push starts and where it ends.

A step that is not tried keeps its tool as it was for its 0.2 s (`_common.idle_command`): closed
when the previous step ends holding the step's object (`hold_after`), open otherwise. An open tool
used to drop the held part: a frame holding a panel 37 mm up, whose next step starts the clamp out
of reach, let the panel fall back onto its rails.

The margin is 5 mm (`_common.REACH_MARGIN` [assumed], the twist's and the insert's) everywhere but
on a lid's lip. The 12 mm of the pinch is its 10 mm yield along the tilted tool plus the 2 mm aim
error; a straight-down tool has no such yield. The panel rail's reach envelope is sized on 5 mm:
its frame lifts the panel with its centre 342.5 mm out (section 13.7), which a 12 mm margin would
refuse. The free moves to a standoff are not checked: a palm held at its travel's end short of the
standoff still meets the face or the feature from its side. The cup's transit above its standoff
on the 110 deg S2e lid is 371 mm out, its standoff 343 mm and the face 318 mm, and it pushes the
lid over as before.

| Case | Before | After |
|---|---|---|
| a 2 x 3 frame lifting a panel at (0, 300 mm), its centre at y 462 mm | `criterion_not_met` after 11.5 s | `out_of_reach` in 0.2 s, y 462.3 mm |
| the gripper on the same panel | silent failure | `out_of_reach`, y 462.3 mm |
| a door at (0, 300 mm, 180 deg) pulled 200 mm open (toward +y) | `slip` | `out_of_reach`, the pull's end at y 514.8 mm |
| the door at (0, -300 mm) left 200 mm open, pushed shut | `criterion_not_met` after 22 s, 0 N | `out_of_reach`, the face at y -460 mm or further |
| N8: the servo gripper's hook on the 240 mm lip lid at y 150 mm, 20 N force goal | `slip`, the palm commanded to 368.7 mm for 865 ticks | `out_of_reach`, the pull on's first pose past 338 mm at y 341.3 mm |

### 14.3 A button the limit cannot actuate

A push presses a button with at least 1.5 x its actuation force, within 80 % of the step's limit.
Where that cap is under the button's actuation force, the press used to run its whole budget
(8.35 s from the front, 5.87 s from the top) and fail silently. Such a press is not tried: it fails
`criterion_not_met` in 0.2 s with `press_cap_n`, `actuation_force_n` and the reason ("The step's
force limit allows 24 N; this button needs at least 25 N to actuate, so it was not pressed": a
25 N button at a 30 N limit).

A cap at or over the actuation force is tried, and the physics decides. The latch counts a call
0.2 mm past the actuation point (about 1.07 x the actuation force on a 3 mm travel, plus the
latch's 0.05 N), but how close a press gets to that depends on the mount. On the wall, pressed
from the front, the press goes 0.27 to 0.32 N over its cap. On the desk, pressed from the top, it
goes 0.42 to 0.54 N over and the button latches below that force. Measured with the adaptive
gripper (cap, then the result):

| Button | Limit | Cap | Wall, from the front | Desk, from the top |
|---|---|---|---|---|
| 8 N | 10 N | 8.0 N | fails, 8.36 N peak | latches, 8.42 N |
| 15 N | 18 N | 14.4 N | not tried (fails on base, 14.73 N) | not tried (fails on base, 14.92 N) |
| 15 N | 19 N | 15.2 N | fails, 15.48 N | latches, 15.69 N |
| 15 N | 20 N | 16.0 N | latches, 16.32 N | latches, 16.48 N |
| 25 N | 30 N | 24.0 N | not tried (fails on base, 24.32 N) | |
| 25 N | 32 N | 25.6 N | fails, 25.86 N | latches, 26.06 N |
| 25 N | 33 N | 26.4 N | fails, 26.67 N | latches, 26.94 N |
| 25 N | 34 N | 27.2 N | latches, 27.50 N | latches, 27.74 N |

Every tried point runs exactly as on base (the same `traj_sha`); only the three caps under the
actuation force are refused, and each of them fails on base too. The borderline wall points are
tried and fail slowly, as on base.

### 14.4 Unchanged runs

`traj_sha` before (4aa5e96) and after, on the S2 gate (7 primitives, the reference and the wrong
tool, 14 conditions each), the H1 lane (8 conditions, 4 steps), the S2b lip pinch and hook, the
lidded tote, the S2f set-down picks, the template cases (button, door, panel, tile, reach), the
drawers and presses, the lids pushed shut from 110 deg and the N8 lids: 268 of the 272 runs are
identical step for step; the four whose push or pull this change is for: the door shut after the load at
3 N and at 10 N (both shut now to 3.9 mm), the S2d drawer on seed 3, and the N8 hook (now
`out_of_reach`) changed, as intended.

`rules.THREADED_TEMPLATES` is now derived from `types.THREAD_KINDS` (one set).
