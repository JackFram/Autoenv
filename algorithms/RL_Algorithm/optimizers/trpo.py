import scipy.optimize
from algorithms.RL_Algorithm.optimizers.utils import *


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = zeros(b.size(), device=b.device).double()
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve
        print("stepfrac: {}, actual improve: {}, expected improve: {}, accept ratio: {}, actual ratio: {}".format(
            stepfrac,
            actual_improve,
            expected_improve,
            accept_ratio,
            ratio
        ))

        if ratio > accept_ratio:
            return True, x_new
    return False, x


def trpo_step(policy_net, states, actions, advantages, max_kl, damping, use_fim=True):

    # """update critic"""
    # def get_value_loss(flat_params):
    #     set_flat_params_to(value_net, tensor(flat_params).double())
    #     for param in value_net.parameters():
    #         if param.grad is not None:
    #             param.grad.data.fill_(0)
    #     values_pred = value_net.predict({"observations": states})
    #     value_loss = (torch.tensor(values_pred - returns).double()).pow(2).mean()
    #
    #     # weight decay
    #     for param in value_net.parameters():
    #         value_loss += param.pow(2).sum() * l2_reg
    #     print(value_loss)
    #     value_loss.backward()
    #     return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()
    #
    # flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
    #                                                         get_flat_params_from(value_net).detach().cpu().numpy(),
    #                                                         maxiter=25)
    # print("after optimize: ", tensor(flat_params))
    # set_flat_params_to(value_net, tensor(flat_params))

    """update policy"""
    with torch.no_grad():
        fixed_log_probs = policy_net.get_log_prob(states, actions)
    """define the loss function for TRPO"""
    def get_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            action_loss = -torch.tensor(advantages).flatten() * torch.exp(log_probs - fixed_log_probs)
            return action_loss.mean()

    """use fisher information matrix for Hessian*vector"""
    def Fvp_fim(v):
        M, mu, info = policy_net.get_fim(states, actions)
        mu = mu.view(-1)
        filter_input_ids = set() if policy_net.is_disc_action else {info['std_id']}

        t = ones(mu.size(), requires_grad=True, device=mu.device).double()
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, policy_net.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = torch.autograd.grad(Jtv, t)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, policy_net.parameters(), filter_input_ids=filter_input_ids).detach()
        JTMJv /= states.shape[0]
        if not policy_net.is_disc_action:
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

    """directly compute Hessian*vector from KL"""
    def Fvp_direct(v):
        kl = policy_net.get_kl(states, actions)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping

    Fvp = Fvp_fim if use_fim else Fvp_direct

    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters(), allow_unused=True)
    grads = [grad.contiguous() for grad in grads]
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
    lm = math.sqrt(max_kl / shs)
    fullstep = stepdir * lm
    expected_improve = -loss_grad.dot(fullstep)
    # print("loss: {}, loss_grad: {}, stepdir: {}, shs: {}, lm: {}, fullstep: {}, expected_improve: {}".format(
    #     loss, loss_grad, stepdir, shs, lm, fullstep, expected_improve
    # ))

    prev_params = get_flat_params_from(policy_net)
    print("prev_params: ", prev_params)
    success, new_params = line_search(policy_net, get_loss, prev_params, fullstep, expected_improve)
    print("new_params: ", new_params)
    print("success flag: ", success)
    set_flat_params_to(policy_net, new_params)

    return success