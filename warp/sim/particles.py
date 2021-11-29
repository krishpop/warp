import warp as wp

@wp.func 
def particle_force(n: wp.vec3,
                       v: wp.vec3,
                       c: float,
                       k_n: float,
                       k_d: float,
                       k_f: float,
                       k_mu: float):
    vn = wp.dot(n, v)
    jn = c*k_n
    jd = min(vn, 0.0)*k_d

    # contact force
    fn = jn + jd

    # friction force
    vt = v - n*vn
    vs = wp.length(vt)
    
    if (vs > 0.0):
        vt = vt/vs

    # Coulomb condition
    ft = wp.min(vs*k_f, k_mu*wp.abs(fn))

    # total force
    return  -n*fn - vt*ft


@wp.kernel
def eval_particle_forces_kernel(grid : wp.uint64,
                 particle_x: wp.array(dtype=wp.vec3),
                 particle_v: wp.array(dtype=wp.vec3),
                 particle_f: wp.array(dtype=wp.vec3),
                 radius: float,
                 k_contact: float,
                 k_damp: float,
                 k_friction: float,
                 k_mu: float,
                 k_cohesion: float):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x = particle_x[i]
    v = particle_v[i]

    f = wp.vec3(0.0, 0.0, 0.0)

    # particle contact
    query = wp.hash_grid_query(grid, x, radius+k_cohesion)
    index = int(0)

    while(wp.hash_grid_query_next(query, index)):

        if index != i:
            
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius*2.0

            if (err <= k_cohesion):
                
                n = n/d
                vrel = v - particle_v[index]

                f = f + particle_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)

    particle_f[i] = f


def eval_particle_forces(model, state, forces):

    wp.launch(
        kernel=eval_particle_forces_kernel,
        dim=model.particle_count,
        inputs=[
            model.particle_grid.id,
            state.particle_q,
            state.particle_qd,
            forces,
            model.particle_radius,
            model.particle_ke,
            model.particle_kd,
            model.particle_kf,
            model.particle_mu,
            model.particle_cohesion
        ],
        device=model.device)
