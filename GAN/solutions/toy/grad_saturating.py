def generator_grad(d_fake, saturating):
    """Magnitude of the gradient of the generator loss with respect to the logit.

    `d_fake` is the raw score the discriminator gives to a fake sample, so that
    D(G(z)) = sigmoid(d_fake). Both losses are written so that the generator
    *minimizes* them:

        saturating      L_G =  log(1 - D(G(z))) = -softplus(d_fake)
        non-saturating  L_G = -log D(G(z))      =  softplus(-d_fake)

    Differentiating by hand gives |dL_G/dd| = D for the first and 1 - D for the
    second; this function returns what `autograd` computes, so that the two can be
    compared point by point.
    """
    d = d_fake.clone().detach().requires_grad_(True)
    loss = -F.softplus(d) if saturating else F.softplus(-d)
    grad, = torch.autograd.grad(loss.sum(), d)
    return grad.abs()


# Check against the closed form, away from and inside the saturated regime
d_grid = torch.linspace(-10, 10, 201)
D_grid = torch.sigmoid(d_grid)

auto_sat = generator_grad(d_grid, saturating=True)
auto_ns = generator_grad(d_grid, saturating=False)

print(f"saturating     : max |autograd - D|     = {(auto_sat - D_grid).abs().max():.2e}")
print(f"non-saturating : max |autograd - (1-D)| = {(auto_ns - (1 - D_grid)).abs().max():.2e}")

plt.figure(figsize=(7, 4))
plt.plot(D_grid, auto_sat, label=r"saturating, $|\partial L_G / \partial d| = D$")
plt.plot(D_grid, auto_ns, label=r"non-saturating, $|\partial L_G / \partial d| = 1 - D$")
plt.axvspan(0, 0.1, color="crimson", alpha=0.10)
plt.text(0.105, 0.5, "generator losing", color="crimson", fontsize=9)
plt.xlabel(r"$D(G(z))$, how convincing the fake sample is")
plt.ylabel("gradient magnitude at the logit")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
