# Tarty #

This is [tart](https://tart.run) manager tool.

MacOS has some sad license derived limitations: you can have only 2
virtualized copies of it

So what this does is provide single script orchestrator for up to 2
virtualized Macs, with nightly update of VM to match what is available
on the internet. Each github runner should be ephemereal and run only
one request until it gets recycled.

## Configuration ##

Uses a configuration file containing:

- Github Personal Access Token (PAT)
- Organization
- Repository
- Base tart image name (one from which runner images are generated)
- Runner tart image name (verified earlier to work)
- SSH username (it should work without login)
- Command to run on base tart image to convert it to runner tart image
  (if it fails, runner image is not updated)
- Hour of day (0-23) when nightly updates should run (default: 2)

# Architecture #

## Core Components ##

1. **VM Lifecycle Manager** - Handle creating, starting, stopping, and
   destroying tart VMs

2. **GitHub Runner Registration** - Register runners with GitHub using the PAT
3. **Image Management** - Create ephemeral runner images nightly
4. **Resource Monitor** - Track the 2-VM limit and manage VM allocation

## Implementation Design ##

**Configuration File** - Read PAT, org, repo, and image names from config file

**VM Pool Management** - Maintain up to 2 VMs, ensuring they're
  ephemeral (one job per VM)

**Nightly Updates** - Schedule automatic runner image updates from
  base image. Base image is manually updated. Nightly updates run in
  main orchestration thread to respect 2-VM limit, and kill one VM for
  their duration.

**Runner Lifecycle** - Spin up VM → register runner → runner accepts
  jobs automatically → destroy VM after completion

**Error Handling** - Graceful cleanup if VMs fail or jobs timeout

## Architecture Flow ##

- Main orchestrator loop managing VM lifecycle
- Subprocess management for tart VM operations
- Simple state tracking (VMs - running or not)
- Ephemeral runners automatically accept jobs when available
- Logging for debugging VM and runner issues


# Requirements #

## Library requirements ##

Almost none - pydantic is used for input validation.

## Implementation requirements ##

This should be single file Python script, with unit tests outside it.

## Base image requirements ##

- The base image should be running ssh server.

- Github Actions Runner should be installed in actions-runner
  subdirectory of the admin user (which is default user to be used)
