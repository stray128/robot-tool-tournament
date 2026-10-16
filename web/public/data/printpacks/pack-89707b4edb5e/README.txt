Print pack: fingers for l_bracket
==============================================

Simulated, untested on hardware. These results come from a physics simulation (MuJoCo) of an open-source parallel gripper; print and test before relying on them.

Files
-----
finger_left.stl, finger_right.stl
    The bare fingers in millimetres, in the gripper's assembly frame: the root is on
    the frame's front face (z = 0) and the finger hangs toward -Z; the contact face
    is the plane y = 0. Use these if you attach fingers to your own jaws.
jaw.stl  (print 2)
    Carriage block for the two guide rods (bores 6.1 mm at 32 mm centres),
    fused with the finger, in millimetres and already turned for printing: contact
    face on the bed, rod bores vertical.
    The second jaw is the same part turned 180 degrees about the gripper's vertical axis.
    Drive: not included (its position is not documented); transfer it from a stock jaw.
certificate.html
    The simulated stress-test results; open it in any browser (no internet needed).
design.json, certificate.json, manifest.json
    The frozen design record, the raw test results and checksums of every file.

Printing
--------
PLA or PETG, 0.2 mm layers, 4 perimeters, 40 % infill or more. Ream the rod bores
to size if they print tight. The simulation assumed solid PLA fingers (4.7 g each)
and a friction coefficient of 0.3 between finger and part.

Check before printing
---------------------
Dimensions marked "assumed" in manifest.json (block height, rod axis height and,
if enabled, the rack) are not documented; compare them with a stock jaw first.
The interface dimensions were measured from an open-source parallel gripper's
published hardware sources (CERN-OHL-P-2.0, attribution in the project README).
This model is new; it contains no files from that project.
