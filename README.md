<p align="center">
  <img src="https://raw.githubusercontent.com/pluto/.github/main/profile/assets/assets_ios_Pluto-1024%401x.png" alt="Pluto Logo" width="50" height="50">
  <br>
  <b style="font-size: 24px;">Pluto</b>
</p>
<p align="center">
  <a href="https://t.me/pluto_xyz/1"><img src="https://img.shields.io/badge/Telegram-Group-8B5CF6?style=flat-square&logo=telegram&logoColor=white&labelColor=24292e&scale=1.5" alt="Telegram"></a>
  <a href="https://docs.pluto.xyz/"><img src="https://img.shields.io/badge/Docs-Pluto-8B5CF6?style=flat-square&logo=readme&logoColor=white&labelColor=24292e&scale=1.5" alt="Docs"></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-8B5CF6.svg?label=license&labelColor=2a2f35" alt="License">
</p>

# Custom Constraints

An implementation of the Customizable Constraint System (CCS) to be used in SNARKs.

## Overview

This library provides an ergonomic and performant implementation of CCS to be adapted to any frontend or backend proving system.

## Features

- **Minimal Dependencies**: Most, if not all, is built by Pluto.
- **Frontend Compatibility**: 
  - Noir (coming soon)

## Usage

Add to your `Cargo.toml`:
```toml
[dependencies]
custom-constraints = "*"
```

## Implementation Details

See the [Customizable constraint systems for succinct arguments](https://eprint.iacr.org/2023/552) paper.

## Roadmap

- [x] CSR Sparse matrices
- [ ] CCS structure
- [ ] CCS checking
- [ ] CCS builder/allocator (i.e., from constraints)

## Contributing

We welcome contributions to our open-source projects. If you want to contribute or follow along with contributor discussions, join our main [Telegram channel](https://t.me/pluto_xyz/1) to chat about Pluto's development.

Our contributor guidelines can be found in our [CONTRIBUTING.md](https://github.com/pluto/.github/blob/main/profile/CONTRIBUTING.md).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

## License

This project is licensed under the Apache V2 License - see the [LICENSE](LICENSE) file for details.

## References

- [Customizable constraint systems for succinct arguments](https://eprint.iacr.org/2023/552)

