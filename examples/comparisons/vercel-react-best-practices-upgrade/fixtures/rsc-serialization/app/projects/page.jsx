import { ProjectsClient } from './projects-client.jsx';
import { getProjects } from '../../lib/projects.mjs';

export default async function ProjectsPage() {
  const projects = await getProjects();
  const projectNames = projects.map((project) => project.name);

  return <ProjectsClient projects={projects} projectNames={projectNames} />;
}
