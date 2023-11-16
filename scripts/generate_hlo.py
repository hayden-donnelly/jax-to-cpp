import jax
import jax.numpy as jnp

def target_fn(x):
    return x**2

def main():
    hlo_comp = jax.xla_computation(target_fn)(jnp.ones((1,)))
    hlo_comp_text = hlo_comp.as_hlo_text().strip()
    with open('hlo_modules/hlo_comp.txt', 'w') as f:
        f.write(hlo_comp_text)
    
    print('Saved the following HLO computation:\n')
    print(hlo_comp_text)

if __name__ == '__main__':
    main()