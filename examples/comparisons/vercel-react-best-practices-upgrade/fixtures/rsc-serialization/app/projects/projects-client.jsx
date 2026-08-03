'use client';

export function ProjectsClient({ projects, projectNames }) {
  return (
    <section aria-labelledby="projects-title">
      <h1 id="projects-title">Projects</h1>
      <p>{projectNames.join(', ')}</p>
      <ul>
        {projects.map((project) => (
          <li key={project.id}>{project.name}</li>
        ))}
      </ul>
    </section>
  );
}
