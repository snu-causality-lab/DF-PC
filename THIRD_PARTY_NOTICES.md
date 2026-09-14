# Third-party notices

The root [MIT license](LICENSE) applies to original DF-PC contributions, except where separate notices apply. It does not replace third-party licenses or retroactively relicense historical third-party code.

## Conditional-independence code

[cddd/independence.py](cddd/independence.py) retains a pcalg attribution and `License: GPLv2` notice. This file is excluded from the new MIT grant; its existing notice is unchanged. Adding the MIT license does not resolve the licensing scope of its later wrapper implementation.

The earlier `src/cddd/cit.py`, retained in git history, contains a discrete G-square implementation with the same attribution and a close match to [gsq/discrete.py](https://github.com/keiichishima/gsq/blob/6d9a0298e3aae1ae27384c2567e254c32554260d/gsq/discrete.py). That upstream project identifies Keiichi SHIMA and declares GPLv2-or-later in [setup.py](https://github.com/keiichishima/gsq/blob/6d9a0298e3aae1ae27384c2567e254c32554260d/setup.py); its [license text](https://github.com/keiichishima/gsq/blob/6d9a0298e3aae1ae27384c2567e254c32554260d/LICENSE) and attribution to R pcalg remain relevant to those historical sources.

External dependencies retain their own licenses. This notice is not a complete dependency-license inventory, and the repository should not be described as wholly MIT-licensed.
