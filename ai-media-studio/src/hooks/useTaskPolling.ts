import { useQuery } from "@tanstack/react-query";
import { getTaskResult } from "@/lib/api";

export const useTaskPolling = (taskId: string | null) => {
  return useQuery({
    queryKey: ["task", taskId],
    queryFn: () => getTaskResult(taskId!),
    enabled: !!taskId,
    refetchInterval: (data) =>
      data?.status === "SUCCESS" || data?.status === "FAILURE" ? false : 2000,
  });
};