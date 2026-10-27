"""Read the explicitly supported suspension configuration from an actual echo."""
import re
import math

INDEX=((1,1),(1,2),(2,1),(2,2))


def scalar(text,key):
    rows=re.findall(r'^(?:!\s+)?'+re.escape(key)+r'\s+([-+\d.eE]+)',text,re.M)
    if len(rows)!=1:raise ValueError('Unresolved echo parameter:'+key)
    value=float(rows[0])
    if not math.isfinite(value):raise ValueError('Nonfinite echo parameter:'+key)
    return value


def wheels(text,key):return [scalar(text,f'{key}({a},{b})') for a,b in INDEX]


def parse(text):
    ks=wheels(text,'FS_COMP_COEFFICIENT')
    if ks!=wheels(text,'FS_EXT_COEFFICIENT'):raise ValueError('Different compression/extension spring slopes unsupported')
    expected={'FS_COMP_GAIN':1.,'FS_EXT_GAIN':1.,'FS_COMP_OFFSET':20.,'FS_EXT_OFFSET':-20.,
              'FS_COMP_FSTATIC_CONSTANT':1.,'FS_EXT_FSTATIC_CONSTANT':1.,'FS_OFFSET':0.,
              'FD_GAIN':1.,'FD_OFFSET':0.,'CMP_R_START_FD':0.,'CMP_R_SCALE_FD':1.,
              'CMP_SPR_SEAT_GAIN':1.,'CMP_SPR_SEAT_OFFSET':0.,'JNC_START_CMP_SPR_SEAT':0.,'JNC_SCALE_CMP_SPR_SEAT':1.,
              'CMP_DAMP_GAIN':1.,'CMP_DAMP_OFFSET':0.,'JNC_START_CMP_DAMP':0.,'JNC_SCALE_CMP_DAMP':1.,'OPT_EXT_SP':0.}
    for key,want in expected.items():
        if wheels(text,key)!=[want]*4:raise ValueError('Unsupported source convention:'+key)
    tables=[]
    for a,b in INDEX:
        found=re.findall(r'^'+re.escape(f'FD_TABLE({a},{b})')+r'\s+SPLINE[^\n]*\n(.*?)^ENDTABLE',text,re.M|re.S)
        if len(found)!=1:raise ValueError('One damper SPLINE table required')
        table=[]
        for line in found[0].splitlines():
            cols=line.strip().split(',')
            if len(cols)!=2:raise ValueError('Two-column damper table required')
            table.append([float(x) for x in cols])
        tables.append(table)
    if any(t!=tables[0] for t in tables):raise ValueError('Per-wheel damper tables differ; common-table component unsupported')
    for axle in (1,2):
        for key,want in [('MX_AUX_GAIN',1.),('MX_AUX_OFFSET',0.),('ROLL_START_MX_AUX',0.),
                         ('ROLL_SCALE_MX_AUX',1.),('MX_AUX_FSTATIC_CONSTANT',0.),('DAUX',0.)]:
            if scalar(text,f'{key}({axle})')!=want:raise ValueError('Unsupported antiroll configuration')
    return dict(spring_stiffness_Npm=[x*1000 for x in ks],spring_motion_ratio=wheels(text,'CMP_SPR_SEAT_COEFFICIENT'),
                design_compression_m=[x/1000 for x in wheels(text,'CMP_DESIGN')],
                damper_motion_ratio=wheels(text,'CMP_DAMP_COEFFICIENT'),
                antiroll_Nmprad=[scalar(text,f'MX_AUX_COEFFICIENT({i})')*180/math.pi for i in (1,2)],
                track_m=[scalar(text,f'L_TRACK({i})')/1000 for i in (1,2)],
                damper_rate_mps=[row[0]/1000 for row in tables[0]],damper_force_N=[row[1] for row in tables[0]],
                unsprung_mass_kg=[a+b+c for a,b,c in zip(wheels(text,'M_US_IND'),wheels(text,'M_US_STR'),wheels(text,'M_TIRE'))],
                design_suspension_axle_load_N=[scalar(text,f'FSA_DESIGN({i})') for i in (1,2)],
                scope='Source constitutive parameters with midpoint-spring and linear-damper approximations')
