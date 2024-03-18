import bpy 

def track_to_constraints(obj, target):
    constraint = obj.constraints.new('TRACK_TO')
    constraint.target = target 
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'